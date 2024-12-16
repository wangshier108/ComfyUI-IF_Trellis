import os
import sys
import torch
import imageio
import numpy as np
import logging
import traceback
import uuid
from PIL import Image
import folder_paths
import rembg

# Ensure environment variables are set before other imports
os.environ['SPCONV_ALGO'] = 'native'

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

logger = logging.getLogger("IF_Trellis")

class IF_TrellisImageTo3D:
    """ComfyUI node for converting images to 3D using TRELLIS."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TRELLIS_MODEL",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 12.0, "step": 0.1}),
                "ss_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "slat_guidance_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 12.0, "step": 0.1}),
                "slat_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "mesh_simplify": ("FLOAT", {"default": 0.95, "min": 0.9, "max": 1.0, "step": 0.01}),
                "texture_size": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 512}),
                "fps": ("INT", {"default": 15, "min": 1, "max": 60}),
                "save_glb": ("BOOLEAN", {"default": True}),
                "save_video": ("BOOLEAN", {"default": True}),
                "texture_mode": (["none", "fast"],),
                "project_name": ("STRING", {"default": "trellis_output"})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")  # (glb_path, video_path)
    RETURN_NAMES = ("glb_path", "video_path")
    FUNCTION = "image_to_3d"
    CATEGORY = "ImpactFrames💥🎞️/Trellis"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_output_dir(self, project_name):
        """Ensure output directory exists"""
        out_dir = os.path.join(self.output_dir, project_name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def tensor_to_pil(self, tensor):
        """Convert tensor from ComfyUI format to PIL Image"""
        if len(tensor.shape) == 4:
            tensor = tensor.squeeze(0)  # Remove batch dimension
        if tensor.shape[0] == 3:  # If in CHW format
            tensor = tensor.permute(1, 2, 0)  # Convert to HWC
        
        # Convert to numpy and scale to 0-255
        image_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np, mode='RGB')

    def preprocess_comfy_image(self, tensor):
        """Preprocess image tensor from ComfyUI format"""
        try:
            # Convert tensor to PIL
            pil_image = self.tensor_to_pil(tensor)
            
            # Initialize rembg session if needed
            if getattr(self, 'rembg_session', None) is None:
                logger.info("Initializing rembg session...")
                self.rembg_session = rembg.new_session('u2net')
            
            # Check if image already has transparency
            has_alpha = False
            if pil_image.mode == 'RGBA':
                alpha = np.array(pil_image)[:, :, 3]
                if not np.all(alpha == 255):
                    has_alpha = True
            
            # Remove background if no alpha channel
            if not has_alpha:
                # Convert to RGB and resize if needed for background removal
                pil_image = pil_image.convert('RGB')
                max_size = max(pil_image.size)
                if max_size > 1024:
                    scale = 1024 / max_size
                    pil_image = pil_image.resize(
                        (int(pil_image.width * scale), 
                         int(pil_image.height * scale)), 
                        Image.Resampling.LANCZOS
                    )
                # Remove background
                pil_image = rembg.remove(pil_image, session=self.rembg_session)
            
            # Process alpha channel and find object bounds
            output_np = np.array(pil_image)
            alpha = output_np[:, :, 3]
            
            # Find bounding box of non-transparent pixels
            bbox = np.argwhere(alpha > 0.8 * 255)
            if len(bbox) == 0:  # Handle case where image is completely transparent
                logger.warning("Image is completely transparent after background removal")
                return pil_image
            
            # Calculate tight bounding box
            y_min, x_min = bbox.min(axis=0)
            y_max, x_max = bbox.max(axis=0)
            
            # Calculate center and dimensions
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            
            # Determine crop size maintaining aspect ratio
            target_size = 518  # Trellis expects 518x518
            crop_size = max(width, height) * 1.2  # Add 20% padding
            
            # Calculate crop bounds maintaining aspect ratio
            x1 = int(max(0, center_x - crop_size/2))
            y1 = int(max(0, center_y - crop_size/2))
            x2 = int(min(pil_image.width, center_x + crop_size/2))
            y2 = int(min(pil_image.height, center_y + crop_size/2))
            
            # Crop and resize maintaining aspect ratio
            output = pil_image.crop((x1, y1, x2, y2))
            
            # Calculate resize dimensions maintaining aspect ratio
            aspect = output.width / output.height
            if aspect > 1:  # Wider than tall
                new_width = target_size
                new_height = int(target_size / aspect)
            else:  # Taller than wide
                new_height = target_size
                new_width = int(target_size * aspect)
            
            # Resize maintaining aspect ratio
            output = output.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create square canvas
            square = Image.new('RGBA', (target_size, target_size), (0, 0, 0, 0))
            
            # Paste resized image in center of square canvas
            paste_x = (target_size - new_width) // 2
            paste_y = (target_size - new_height) // 2
            square.paste(output, (paste_x, paste_y))
            
            # Convert to numpy and process alpha
            output_np = np.array(square).astype(np.float32) / 255
            output = output_np[:, :, :3] * output_np[:, :, 3:4]
            
            # Convert back to PIL
            return Image.fromarray((output * 255).astype(np.uint8))
            
        except Exception as e:
            logger.error(f"Error in preprocess_comfy_image: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    @torch.no_grad()
    def image_to_3d(self, model, image, seed=0,
                    ss_guidance_strength=7.5, ss_sampling_steps=12, 
                    slat_guidance_strength=3.0, slat_sampling_steps=12,
                    mesh_simplify=0.95, texture_size=1024, fps=15,
                    save_glb=True, save_video=True, texture_mode="fast", 
                    project_name="trellis_output"):
        """Convert image to 3D representation"""
        
        try:
            # Create output directory
            out_dir = self.ensure_output_dir(project_name)
            
            # Preprocess ComfyUI image tensor
            pil_image = self.preprocess_comfy_image(image)

            # Run pipeline with modified parameters
            formats = ['mesh']  # Always need mesh for GLB
            if save_video:
                formats.append('gaussian')  # Only add gaussian if video needed
            
            outputs = model.run(
                pil_image,
                seed=seed,
                sparse_structure_sampler_params={
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_guidance_strength,
                },
                formats=formats,  # Pass formats list
                preprocess_image=False  # Skip Trellis preprocessing since we did it already
            )

            # Initialize output paths
            glb_path = ""
            video_path = ""

            # Generate and save video preview if requested
            if save_video and 'gaussian' in outputs:
                video_frames = render_utils.render_video(outputs['gaussian'][0])['color']
                video_path = os.path.join(out_dir, f"{project_name}_preview.mp4")
                imageio.mimsave(video_path, video_frames, fps=fps)

            # Export GLB if requested
            if save_glb:
                try:
                    glb = postprocessing_utils.to_glb(
                        outputs.get('gaussian', [None])[0],  # Pass None if gaussian not available
                        outputs['mesh'][0],
                        simplify=mesh_simplify,
                        texture_size=texture_size,
                        texture_mode=texture_mode,
                        fill_holes=True,
                        debug=False,
                        verbose=True,
                    )
                    glb_path = os.path.join(out_dir, f"{project_name}.glb") 
                    glb.export(glb_path)
                except Exception as e:
                    logger.error(f"Error exporting GLB: {str(e)}")
                    logger.error(traceback.format_exc())

            # Return strings directly (not as lists) for ComfyUI
            return (glb_path, video_path)

        except Exception as e:
            logger.error(f"Error in image_to_3d: {str(e)}")
            logger.error(traceback.format_exc())
            raise