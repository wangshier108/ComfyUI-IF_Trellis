# IF_TrellisCheckpointLoader.py
import os
import sys
import importlib
import torch
import logging
import folder_paths
from huggingface_hub import hf_hub_download, snapshot_download
from .trellis_model_manager import get_config, move_checkpoints_to_comfyui

class IF_TrellisCheckpointLoader:
    """
    Node to manage the loading of the TRELLIS model.
    Follows ComfyUI conventions for model management.
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["TRELLIS-image-large"],),
                "use_fp16": ("BOOLEAN", {"default": True}),
                "enable_xformers": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("TRELLIS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "ImpactFrames💥🎞️/Trellis"

    def optimize_pipeline(self, pipeline, use_fp16=True, enable_xformers=True):
        """Apply optimizations to the pipeline if available"""
        if self.device == "cuda":
            try:
                pipeline.cuda()
                
                # Apply FP16 optimization if available and requested
                if use_fp16:
                    if hasattr(pipeline, 'enable_attention_slicing'):
                        pipeline.enable_attention_slicing()
                    if hasattr(pipeline, 'half'):
                        pipeline.half()
                    
                # Apply xformers optimization if available and requested
                if enable_xformers and hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                    pipeline.enable_xformers_memory_efficient_attention()
                    
            except Exception as e:
                logging.warning(f"Some optimizations failed: {str(e)}")
                
        return pipeline

    def load_model(self, model_name, use_fp16=True, enable_xformers=True):
        """
        Load and optimize the TRELLIS model.
        
        Args:
            model_name (str): Name of the model to load
            use_fp16 (bool): Whether to use FP16 optimization
            enable_xformers (bool): Whether to enable xformers optimization
        """
        try:
            # Ensure models are in place
            move_checkpoints_to_comfyui()
            
            # Get model path using ComfyUI's folder structure
            model_path = os.path.join(folder_paths.get_folder_paths("checkpoints")[0], model_name)
            
            if not os.path.exists(model_path):
                raise ValueError(f"Model path not found: {model_path}")
            
            # Load configuration
            config = get_config(model_path)
            if not config:
                raise ValueError(f"Failed to load config from {model_path}")
            
            # Initialize pipeline following app.py pattern
            from trellis.pipelines import TrellisImageTo3DPipeline
            pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
            
            # Apply optimizations
            pipeline = self.optimize_pipeline(pipeline, use_fp16, enable_xformers)
            
            return (pipeline,)
            
        except Exception as e:
            logging.error(f"Error loading TRELLIS model: {str(e)}")
            raise