#IF_Preview_3D.py
import os
import logging
import folder_paths
import traceback

class IF_Preview_3D:
    """
    Node to display the 3D model (GLB).
    Integrates with Three.js viewer in the frontend.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_file": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "Path to .glb file"
                }),
            },
        }
    
    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "ImpactFrames💥🎞️/Trellis"
    
    def validate_file(self, file_path: str, expected_ext: str) -> str:
        """Validate file exists and has correct extension"""
        if not file_path:
            return ""
            
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            return ""
            
        ext = os.path.splitext(file_path)[1].lower()
        if ext != expected_ext:
            logging.warning(f"Invalid file extension for {file_path}. Expected {expected_ext}")
            return ""
            
        return file_path

    def preview(self, glb_file: str):
        """Preview the 3D model"""
        try:
            # Validate file
            glb_path = self.validate_file(glb_file, '.glb')
            
            return {
                "mesh": [glb_path] if glb_path else None
            }
        except Exception as e:
            logging.error(f"Error in preview: {str(e)}")
            return {}

    @classmethod
    def IS_CHANGED(cls, glb_file: str) -> str:
        """
        Check if preview file has changed
        
        Returns:
            str: Timestamp string for change detection
        """
        return str(os.path.getmtime(glb_file) if glb_file and os.path.exists(glb_file) else 0)