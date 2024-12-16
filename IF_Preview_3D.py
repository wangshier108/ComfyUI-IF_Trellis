#IF_Preview_3D.py
import os
import folder_paths
import nodes

def normalize_path(path):
    return path.replace('\\', '/')

class IF_Preview_3D:
    @classmethod
    def INPUT_TYPES(s):
        # Get paths for both input and output directories since GLB files could be in either
        input_dir = os.path.join(folder_paths.get_input_directory(), "3d")
        output_dir = folder_paths.get_output_directory()
        
        # Create 3d input directory if it doesn't exist
        os.makedirs(input_dir, exist_ok=True)

        # Get GLB files from both input and output directories
        input_files = [normalize_path(os.path.join("3d", f)) for f in os.listdir(input_dir) if f.endswith('.glb')]
        output_files = [normalize_path(os.path.join("output", f)) for f in os.listdir(output_dir) if f.endswith('.glb')]
        
        # Combine and sort all GLB files
        all_files = sorted(input_files + output_files)
        
        if len(all_files) == 0:
            all_files = ["None.glb"]  # Provide a default option if no files found

        return {"required": {
            "model_file": (all_files, {"file_upload": True}),
            "glb_path": ("STRING", {"default": None}),  # New input for GLB path from IF_Trellis
            "image": ("LOAD_3D", {}),
            "width": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
            "height": ("INT", {"default": 1024, "min": 1, "max": 4096, "step": 1}),
            "show_grid": ([True, False],),
            "camera_type": (["perspective", "orthographic"],),
            "view": (["front", "right", "top", "isometric"],),
            "material": (["original", "normal", "wireframe", "depth"],),
            "bg_color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
            "light_intensity": ("INT", {"default": 10, "min": 1, "max": 20, "step": 1}),
            "up_direction": (["original", "-x", "+x", "-y", "+y", "-z", "+z"],),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "mesh_path")
    FUNCTION = "process"
    CATEGORY = "ImpactFrames💥🎞️/Trellis"

    def process(self, model_file, glb_path, image, **kwargs):
        # If glb_path is provided and exists, use it instead of model_file
        filepath = None
        if glb_path and os.path.exists(glb_path):
            filepath = glb_path
        else:
            # Handle paths from both input and output directories
            if model_file.startswith("output/"):
                filepath = os.path.join(folder_paths.get_output_directory(), model_file[7:])
            else:
                filepath = folder_paths.get_annotated_filepath(model_file)

        # Use the LoadImage node to handle the rendered image
        load_image_node = nodes.LoadImage()
        imagepath = folder_paths.get_annotated_filepath(image)
        
        output_image, output_mask = load_image_node.load_image(image=imagepath)
        
        # Return the absolute path to the GLB file for the renderer
        return output_image, output_mask, filepath

# Node class mappings for registration
NODE_CLASS_MAPPINGS = {
    "IF_Preview_3D": IF_Preview_3D
}

# Display name mappings for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_Preview_3D": "Trellis 3D Preview 🎲🔍"
}