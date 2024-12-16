# IF_TrellisCheckpointLoader.py
import os
import sys
import importlib
import torch
import logging
import folder_paths
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path
import json

class IF_TrellisCheckpointLoader:
    """
    Node to manage the loading of the TRELLIS model.
    Follows ComfyUI conventions for model management.
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_repo = "JeffreyXiang/TRELLIS-image-large"
        
        # Setup paths properly
        self.base_path = Path(folder_paths.get_folder_paths("checkpoints")[0])
        self.model_path = self.base_path / "TRELLIS-image-large"
        self.ckpts_path = self.model_path / "ckpts"

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

    def ensure_model_exists(self):
        """Ensure model files exist, downloading if necessary"""
        try:
            # Create directories if they don't exist
            self.model_path.mkdir(parents=True, exist_ok=True)
            self.ckpts_path.mkdir(parents=True, exist_ok=True)

            # List files in model directory
            #print("\nFiles in model directory:")
            '''for root, dirs, files in os.walk(self.model_path):
                #print(f"\nIn {root}:")
                for f in files:
                    #print(f"  {f}")'''

            # Check if model needs to be downloaded
            config_file = self.model_path / "pipeline.json"
            
            if not config_file.exists() or not any(self.ckpts_path.glob("*.safetensors")):
                logging.info(f"Downloading TRELLIS model from {self.model_repo}")
                
                # Download to a temporary directory first
                temp_dir = self.base_path / "temp_download"
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                logging.info(f"Downloading to temporary directory: {temp_dir}")
                snapshot_download(
                    repo_id=self.model_repo,
                    local_dir=str(temp_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    max_workers=4
                )
                
                # List all downloaded files
                logging.info("Downloaded files:")
                for file in temp_dir.rglob("*"):
                    logging.info(f"Found file: {file}")
                
                # Move files to correct locations
                for src in temp_dir.rglob("*"):
                    if src.is_file():  # Only process files, not directories
                        # Get relative path from temp_dir
                        rel_path = src.relative_to(temp_dir)
                        # Construct destination path
                        dst = self.model_path / rel_path
                        # Ensure parent directory exists
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        # Move file
                        logging.info(f"Moving {rel_path} to {dst}")
                        if dst.exists():
                            dst.unlink()  # Remove existing file if it exists
                        src.replace(dst)
                
                # Cleanup
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(str(temp_dir))
                
                logging.info("Model download complete")
                
                # List files after download
                '''print("\nFiles after download:")
                for root, dirs, files in os.walk(self.model_path):
                    print(f"\nIn {root}:")
                    for f in files:
                        print(f"  {f}")'''
                
                # Verify essential files exist
                if not config_file.exists():
                    raise FileNotFoundError(f"pipeline.json not found after download")
                
                # Check if any .safetensors files exist in ckpts directory
                safetensors_files = list(self.ckpts_path.glob("*.safetensors"))
                if not safetensors_files:
                    raise FileNotFoundError(f"No model files found in {self.ckpts_path}")
                
                # Verify pipeline.json content
                try:
                    with open(config_file, 'r') as f:
                        pipeline_config = json.load(f)
                    #print("\nPipeline config:")
                    #print(json.dumps(pipeline_config, indent=2))
                except Exception as e:
                    print(f"Error reading pipeline.json: {e}")
                
        except Exception as e:
            logging.error(f"Error ensuring model exists: {str(e)}")
            raise

    def optimize_pipeline(self, pipeline, use_fp16=True, enable_xformers=True):
        """Apply optimizations to the pipeline if available"""
        if self.device == "cuda":
            try:
                pipeline.cuda()
                
                if use_fp16:
                    if hasattr(pipeline, 'enable_attention_slicing'):
                        pipeline.enable_attention_slicing()
                    if hasattr(pipeline, 'half'):
                        pipeline.half()
                    
                if enable_xformers and hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                    pipeline.enable_xformers_memory_efficient_attention()
                    
            except Exception as e:
                logging.warning(f"Some optimizations failed: {str(e)}")
                
        return pipeline

    def load_model(self, model_name, use_fp16=True, enable_xformers=True):
        """
        Load and optimize the TRELLIS model.
        """
        try:
            # Ensure model exists
            self.ensure_model_exists()
            
            print(f"Model path after ensuring existence: {self.model_path}")
            print(f"Path exists: {os.path.exists(str(self.model_path))}")
            
            # Initialize pipeline
            from trellis.pipelines import TrellisImageTo3DPipeline
            
            pipeline = TrellisImageTo3DPipeline.from_pretrained(str(self.model_path))
            
            # Apply optimizations
            pipeline = self.optimize_pipeline(pipeline, use_fp16, enable_xformers)
            
            return (pipeline,)
            
        except Exception as e:
            logging.error(f"Error loading TRELLIS model: {str(e)}")
            raise