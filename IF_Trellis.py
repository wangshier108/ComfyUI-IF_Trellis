# IF_Trellis.py
import os
import torch
import imageio
import hashlib
import numpy as np
import logging
import traceback
import uuid
from PIL import Image
import folder_paths
from typing import List, Union, Tuple, Literal, Optional, Dict
from easydict import EasyDict as edict
import gc
import comfy.model_management
import trimesh
import trimesh.exchange.export

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.representations import Gaussian, MeshExtractResult
from comfybridge.protocol.oss_client import OssClient


# OSS Client 实例
client = OssClient(menu="bizyair", is_production=False)

logger = logging.getLogger("IF_Trellis")

def get_subpath_after_dir(full_path: str, target_dir: str) -> str:
    try:
        full_path = os.path.normpath(full_path)
        full_path = full_path.replace('\\', '/')
        path_parts = full_path.split('/')
        try:
            index = path_parts.index(target_dir)
            subpath = '/'.join(path_parts[index + 1:])
            return subpath
        except ValueError:
            return path_parts[-1]
    except Exception as e:
        print(f"Error processing path in get_subpath_after_dir: {str(e)}")
        return os.path.basename(full_path)

def get_sha256_filename(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def upload_with_retries(file_path, max_retries=3):
    upload_data = None
    with open(file_path, "rb") as file:
        upload_data = file.read()
    
    filename = get_sha256_filename(file_path)
    
    attempt = 0
    while attempt < max_retries:
        try:
            url = client.upload_byte(file_name=filename, image_stream=upload_data)
            if url:
                return url
            else:
                logger.error(f"Failed to upload {file_path}. Retrying... ({attempt + 1}/{max_retries})")
        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {e}")
        
        attempt += 1
    
    # logger.error(f"Failed to upload {file_path} after {max_retries} attempts.")
    return None


def upload_to_oss(abs_filename):
    if not os.path.exists(abs_filename):
        print(f"File {abs_filename} does not exist.")
        return
    
    url = upload_with_retries(abs_filename)
    
    if url is not None:
        print(f"File with absolute path uploaded successfully! Access it here: {url}")
        try:
            os.remove(abs_filename)
            print(f"Successfully deleted file {abs_filename}")
        except Exception as e:
            logger.error(f"Failed to delete file {abs_filename}: {e}")
    else:
        print(f"Failed to upload file {abs_filename} after retries.")

    return url



class IF_TrellisImageTo3D:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TRELLIS_MODEL",),
                "mode": (["single", "multi"], {"default": "single", "tooltip": "Mode. single is a single image. with multi you can provide multiple reference angles for the 3D model"}),
                "images": ("IMAGE", {"list": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "ss_guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 12.0, "step": 0.1}),
                "ss_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "slat_guidance_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 12.0, "step": 0.1}),
                "slat_sampling_steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "multimode": (["stochastic", "multidiffusion"], {"default": "stochastic"}),
                },
            "optional": {
                "masks": ("MASK", {"list": True}),
            }
        }

    RETURN_TYPES = ("trellis_gaussian", "trellis_mesh", )
    FUNCTION = "image_to_3d"
    CATEGORY = "ImpactFrames💥🎞️/Trellis"
    OUTPUT_NODE = True

    def __init__(self, vertices=None, faces=None, uvs=None, face_uvs=None, albedo=None):
        self.logger = logger
        self.output_dir = folder_paths.get_output_directory()
        self.temp_dir = folder_paths.get_temp_directory()
        self.device = None
        self.vertices = vertices
        self.faces = faces
        self.uvs = uvs
        self.face_uvs = face_uvs
        self.albedo = albedo
        self.normals = None

    def torch_to_pil_batch(self, images: Union[torch.Tensor, List[torch.Tensor]],
                          masks: Optional[torch.Tensor] = None,
                          alpha_min: float = 0.1) -> List[Image.Image]:
        if isinstance(images, list):
            processed_tensors = []
            for img in images:
                if img.ndim == 3:
                    processed_tensors.append(img)
                elif img.ndim == 4:
                    processed_tensors.extend([t for t in img])
            images = torch.stack(processed_tensors, dim=0)

        logger.info(f"torch_to_pil_batch input shape: {images.shape}")
        if images.ndim == 3:
            images = images.unsqueeze(0)

        if images.shape[-1] != 3:
            if images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1)

        processed_images = []
        for i in range(images.shape[0]):
            img = images[i].detach().cpu()
            if masks is not None:
                if isinstance(masks, torch.Tensor):
                    mask = masks[i] if i < masks.shape[0] else masks[0]
                    if mask.ndim > 2:
                        mask = mask.squeeze()
                    if mask.shape != img.shape[:2]:
                        import torch.nn.functional as F
                        mask = F.interpolate(
                            mask.unsqueeze(0).unsqueeze(0),
                            size=img.shape[:2],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                    if torch.any(mask > alpha_min):
                        mask = mask.to(dtype=img.dtype)
                        mask = mask.unsqueeze(-1) if mask.ndim == 2 else mask
                        img = torch.cat([img, mask], dim=-1)
                        mode = "RGBA"
                    else:
                        mode = "RGB"
                else:
                    mode = "RGB"
            else:
                mode = "RGB"
            img_np = (img.numpy() * 255).astype(np.uint8)
            processed_images.append(Image.fromarray(img_np, mode=mode))
            logger.info(f"Processed image {i}, shape: {img_np.shape}, mode: {mode}")

        return processed_images

    def pack_state(self, gaussian, mesh) -> Dict[str, Dict[str, np.ndarray]]:
        return {
            'gaussian': {
                **gaussian.init_params,
                '_xyz': gaussian._xyz.cpu().numpy(),
                '_features_dc': gaussian._features_dc.cpu().numpy(),
                '_scaling': gaussian._scaling.cpu().numpy(),
                '_rotation': gaussian._rotation.cpu().numpy(),
                '_opacity': gaussian._opacity.cpu().numpy(),
            },
            'mesh': {
                'vertices': mesh.vertices.cpu().numpy(),
                'faces': mesh.faces.cpu().numpy(),
            },
        }

    def unpack_state(self, state: dict) -> Tuple[Gaussian, MeshExtractResult]:
        gaussian = Gaussian(
            aabb=state['gaussian']['aabb'],
            sh_degree=state['gaussian']['sh_degree'],
            mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
            scaling_bias=state['gaussian']['scaling_bias'],
            opacity_bias=state['gaussian']['opacity_bias'],
            scaling_activation=state['gaussian']['scaling_activation'],
        )
        gaussian._xyz = torch.tensor(state['gaussian']['_xyz'], device=self.device)
        gaussian._features_dc = torch.tensor(state['gaussian']['_features_dc'], device=self.device)
        gaussian._scaling = torch.tensor(state['gaussian']['_scaling'], device=self.device)
        gaussian._rotation = torch.tensor(state['gaussian']['_rotation'], device=self.device)
        gaussian._opacity = torch.tensor(state['gaussian']['_opacity'], device=self.device)

        mesh = edict(
            vertices=torch.tensor(state['mesh']['vertices'], device=self.device),
            faces=torch.tensor(state['mesh']['faces'], device=self.device),
        )
        return gaussian, mesh

    def get_pipeline_params(self, seed, ss_sampling_steps, ss_guidance_strength,
                            slat_sampling_steps, slat_guidance_strength):
        if ss_sampling_steps < 1:
            raise ValueError("ss_sampling_steps must be >= 1")
        if slat_sampling_steps < 1:
            raise ValueError("slat_sampling_steps must be >= 1")
        if ss_guidance_strength < 0:
            raise ValueError("ss_guidance_strength must be >= 0")
        if slat_guidance_strength < 0:
            raise ValueError("slat_guidance_strength must be >= 0")

        return {
            "seed": seed,
            "formats": ["gaussian", "mesh"],
            "preprocess_image": True,
            "sparse_structure_sampler_params": {
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            "slat_sampler_params": {
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            }
        }

    @torch.inference_mode()
    def image_to_3d(
        self,
        model: TrellisImageTo3DPipeline,
        mode: str,
        images: torch.Tensor,
        seed: int,
        ss_guidance_strength: float,
        ss_sampling_steps: int,
        slat_guidance_strength: float,
        slat_sampling_steps: int,
        multimode: str,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[str, str, torch.Tensor]:
        try:
            logger.info(f"Input images tensor initial shape: {images.shape}")
            with model.inference_context():
                self.device = model.device

                pipeline_params = self.get_pipeline_params(
                    seed, ss_sampling_steps, ss_guidance_strength,
                    slat_sampling_steps, slat_guidance_strength
                )

                # Handle single vs multi mode differently
                if mode == "single":
                    # Take just the first image regardless of how many were input
                    images = images[0:1]
                    pil_imgs = self.torch_to_pil_batch(images, masks)
                    outputs = model.run(pil_imgs[0], **pipeline_params)
                else:
                    # In multi mode, treat the whole list as a batch
                    pil_imgs = self.torch_to_pil_batch(images, masks)
                    logger.info(f"Processing {len(pil_imgs)} views for multi-view reconstruction")
                    outputs = model.run_multi_image(
                        pil_imgs,
                        mode=multimode,
                        **pipeline_params
                    )
                return outputs['gaussian'], outputs['mesh']

        except Exception as e:
            logger.error(f"Error in image_to_3d: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        finally:
            pass

    def cleanup_outputs(self, outputs):
        # Now we only need to clean up the dictionary itself
        del outputs
        gc.collect()


class Trans3D2Video:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trellis_gaussian": ("trellis_gaussian",  {"forceInput":True}),
                "trellis_mesh": ("trellis_mesh",  {"forceInput":True}),
                "fps": ("INT", {"default": 15, "min": 1, "max": 60, "tooltip": "FPS. the higher the value the smoother the video will be"}),
                "render_video": ("BOOLEAN", {"default": False, "tooltip": "Render a video"}),           
            }
        }

    CATEGORY = "ImpactFrames💥🎞️/Trellis"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)

    def main(self, trellis_gaussian, trellis_mesh, fps, render_video):
        out_dir = os.path.join(folder_paths.get_output_directory(), "trellis_output")
        os.makedirs(out_dir, exist_ok=True)
        video_path = None
        url = None
        gaussian_output = trellis_gaussian[0]
        mesh_output = trellis_mesh[0]
        if render_video:
            video_gs = render_utils.render_video(gaussian_output)['color']
            video_mesh = render_utils.render_video(mesh_output)['normal']
            video = [np.concatenate([frame_gs, frame_mesh], axis=1)
                     for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
            video_path = os.path.join(out_dir, "trellis_output_preview.mp4")
            imageio.mimsave(video_path, video, fps=fps)
            full_video_path = os.path.abspath(video_path)
            logger.info(f"Full video path: {full_video_path}, Processed video path: {video_path}")
            url = upload_to_oss(full_video_path)  

        return (url,)

class Trans3D2GlbFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trellis_gaussian": ("trellis_gaussian",  {"forceInput":True}),
                "trellis_mesh": ("trellis_mesh",  {"forceInput":True}),
                "mesh_simplify": ("FLOAT", {"default": 0.95, "min": 0.9, "max": 1.0, "step": 0.01, "tooltip": "Simplify the mesh. the lower the value more polygons the mesh will have"}),
                "texture_size": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 512, "tooltip": "Texture size. the higher the value the more detailed the texture will be"}),
                "texture_mode": (["blank", "fast", "opt"], {"default": "fast", "tooltip": "Texture mode. blank is no texture. fast is a fast texture. opt is a high quality texture"}),
                "save_glb": ("BOOLEAN", {"default": True, "tooltip": "Save the GLB file this is the 3D model"}),
                "save_texture": ("BOOLEAN", {"default": False, "tooltip": "Save the texture file"}),
            }
        }

    CATEGORY = "ImpactFrames💥🎞️/Trellis"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("url", "texture_image")

    def generate_outputs(self, trellis_gaussian, trellis_mesh, save_glb=True, texture_mode='blank', texture_size=1024, mesh_simplify=0.95, save_texture=True, save_wireframe=False):
        project_name = "trellis_output"
        glb_path = ""
        texture_path = wireframe_path = ""
        texture_image = wireframe_image = None
        gaussian_output = trellis_gaussian[0]
        mesh_output = trellis_mesh[0]

        out_dir = os.path.join(folder_paths.get_output_directory(), project_name)
        os.makedirs(out_dir, exist_ok=True)
          
        if save_glb:
            texture_path = os.path.join(out_dir, f"{project_name}_texture_{uuid.uuid4()}.png") if save_texture else None
            wireframe_path = os.path.join(out_dir, f"{project_name}_wireframe_{uuid.uuid4()}.png") if save_wireframe else None
            glb_path = os.path.join(out_dir, f"{project_name}_{uuid.uuid4()}.glb")

            glb = postprocessing_utils.to_glb(
                gaussian_output,
                mesh_output,
                simplify=mesh_simplify,
                texture_size=texture_size,
                texture_mode=texture_mode,
                fill_holes=True,
                save_texture=save_texture and texture_mode != 'blank',
                texture_path=texture_path,
                save_wireframe=save_wireframe and texture_mode != 'blank',
                wireframe_path=wireframe_path,
                verbose=True
            )
            glb.export(glb_path)
            full_glb_path = os.path.abspath(glb_path)
            logger.info(f"Full GLB path: {full_glb_path}, Processed GLB path: {glb_path}")

            # Handle texture image creation
            if save_texture and texture_mode != 'blank' and texture_path and os.path.exists(texture_path):
                try:
                    texture_image = Image.open(texture_path).convert('RGB')
                    texture_image = np.array(texture_image)
                except Exception as e:
                    logger.warning(f"Failed to load texture image: {str(e)}")
                    texture_image = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
            else:
                # Create a blank texture if not saving or if texture mode is blank
                texture_image = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)

            # Handle wireframe image
            if wireframe_path and os.path.exists(wireframe_path):
                wireframe_image = Image.open(wireframe_path).convert('RGB')
                wireframe_image = np.array(wireframe_image)
            else:
                wireframe_image = None
        
        logger.info(f"Texture image shape: {texture_image.shape}")

        return full_glb_path, glb_path, texture_path, wireframe_path, texture_image, wireframe_image


    def main(
        self,
        trellis_gaussian, 
        trellis_mesh,
        mesh_simplify,
        texture_size,
        texture_mode,
        save_glb,
        save_texture):
        full_glb_path, glb_path, _, _, texture_image, _ = self.generate_outputs(
            trellis_gaussian,
            trellis_mesh,
            save_glb=save_glb,
            texture_mode=texture_mode,
            texture_size=texture_size,
            mesh_simplify=mesh_simplify,
            save_texture=save_texture,
            save_wireframe=False
        )
        # Convert texture image to tensor
        if isinstance(texture_image, np.ndarray):
            # Ensure proper shape and type
            if texture_image.ndim == 2:  # Grayscale
                texture_image = np.stack([texture_image]*3, axis=-1)
            elif texture_image.shape[-1] == 4:  # RGBA
                texture_image = texture_image[..., :3]  # Drop alpha channel
            
            texture_tensor = torch.from_numpy(texture_image).float() / 255.0
            texture_tensor = texture_tensor.unsqueeze(0)  # [1, H, W, 3]
            logger.info(f"Texture tensor shape after unsqueeze: {texture_tensor.shape}")
        else:
            # Fallback to black texture
            texture_tensor = torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32)

        url = upload_to_oss(full_glb_path)
        return (url, texture_tensor)

class Trans3D2Gaussian:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trellis_gaussian": ("trellis_gaussian",  {"forceInput":True}),
                "save_gaussian": ("BOOLEAN", {"default": False, "tooltip": "Save the Gaussian file this is a ply file of the 3D model"}),
            }
        }

    CATEGORY = "ImpactFrames💥🎞️/Trellis"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING", )
    # RETURN_NAMES = ("gaussian_path",)

    def main(
        self,
        trellis_gaussian, 
        save_gaussian):
        gaussian_path = ""
        project_name = "trellis_output"
        if save_gaussian:
            gaussian_path = os.path.join(folder_paths.get_output_directory(), project_name, f"{project_name}.ply")
            trellis_gaussian[0].save_ply(gaussian_path)
        
        url = upload_to_oss(gaussian_path)
        return (url, )


