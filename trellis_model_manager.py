# trellis_model_manager.py
import os
import sys
import logging
import folder_paths
from huggingface_hub import hf_hub_download, snapshot_download
from typing import Dict, Union
import json

# Use logger instead of print for better logging
logger = logging.getLogger('model_manager')

def move_checkpoints_to_comfyui():
    """
    Move model files to the specified ComfyUI folder structure
    """
    try:
        # Get paths using ComfyUI's folder_paths module
        checkpoints_dir = folder_paths.get_folder_paths("checkpoints")[0]
        target_folder = os.path.join(checkpoints_dir, "TRELLIS-image-large")
        ckpts_folder = os.path.join(target_folder, "ckpts")
        
        # Create target directories
        os.makedirs(target_folder, exist_ok=True)
        os.makedirs(ckpts_folder, exist_ok=True)
        
        # Check if models need to be downloaded
        if not os.path.exists(os.path.join(target_folder, "pipeline.json")):
            logger.info("Downloading TRELLIS models...")
            try:
                # Download main pipeline files
                snapshot_download(
                    repo_id="JeffreyXiang/TRELLIS-image-large",
                    local_dir=target_folder,
                    local_dir_use_symlinks=False,
                    allow_patterns=["pipeline.json", "README.md"]
                )
                # Download checkpoint files
                snapshot_download(
                    repo_id="JeffreyXiang/TRELLIS-image-large",
                    local_dir=ckpts_folder,
                    local_dir_use_symlinks=False,
                    allow_patterns=["*.safetensors", "*.json"],
                    cache_dir=os.path.join(target_folder, ".cache")
                )
                logger.info("Model files downloaded successfully")
            except Exception as e:
                logger.error(f"Error downloading model files: {str(e)}")
                raise
    except Exception as e:
        logger.error(f"Error in move_checkpoints_to_comfyui: {str(e)}")
        raise

def load_json(file_path: str) -> Dict:
    """
    Load a JSON configuration file.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return {}

def get_config(path: str) -> Dict:
    """
    Load a configuration file from a model folder or a Hugging Face model hub.

    Args:
        path: The path to the model. Can be either local path or a Hugging Face model name.
        
    Returns:
        Dict: Configuration dictionary with model settings
    """
    try:
        # Check for local pipeline.json
        config_path = os.path.join(path, "pipeline.json")
        
        if os.path.exists(config_path):
            config = load_json(config_path)
        else:
            # Try downloading from HuggingFace
            config_path = hf_hub_download(
                repo_id=f"JeffreyXiang/{os.path.basename(path)}", 
                filename="pipeline.json",
                cache_dir=os.path.join(path, ".cache")
            )
            config = load_json(config_path)
            
        if not config:
            raise ValueError(f"Could not load valid configuration from {path}")
            
        # Add default pipeline name if missing
        if 'name' not in config:
            config['name'] = 'TrellisImageTo3DPipeline'
            
        return config
        
    except Exception as e:
        logger.error(f"Error loading config from {path}: {e}")
        # Return minimal default config
        return {
            'name': 'TrellisImageTo3DPipeline',
            'version': '1.0'
        }