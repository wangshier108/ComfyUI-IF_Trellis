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
from trellis_model_manager import TrellisModelManager
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
from trellis.modules import set_attention_backend
from typing import Literal
from trellis.modules.attention_utils import enable_sage_attention, disable_sage_attention

logger = logging.getLogger("IF_Trellis")

def set_backend(backend: Literal['spconv', 'torchsparse']):
    # Example helper if you wish to call the underlying global set_backend from trellis.modules.sparse:
    from trellis.modules.sparse import set_backend as _set_sparse_backend
    # Also handle spconv algo if desired, e.g. os.environ['SPCONV_ALGO'] = ...
    _set_sparse_backend(backend)

class TrellisConfig:
    """Global configuration for Trellis"""
    def __init__(self):
        self.attention_backend = "sage"
        self.spconv_algo = "implicit_gemm"
        self.smooth_k = True
        self.device = "cuda"
        self.use_fp16 = True
        # Added new configuration dictionary
        self._config = {
            "dinov2_size": "large",  # Default model size
            "dinov2_model": "dinov2_vitg14"  # Default model name
        }
        
    # Added new methods
    def get(self, key, default=None):
        """Get configuration value with fallback"""
        return self._config.get(key, default)
        
    def set(self, key, value):
        """Set configuration value"""
        self._config[key] = value
        
    def setup_environment(self):
        """Set up all environment variables and backends"""
        import os
        from trellis.modules import set_attention_backend
        from trellis.modules.sparse import set_backend
        
        # Set attention backend
        set_attention_backend(self.attention_backend)
        
        # Set smooth k for sage attention
        os.environ['SAGEATTN_SMOOTH_K'] = '1' if self.smooth_k else '0'
        
        # Set spconv algorithm
        os.environ['SPCONV_ALGO'] = self.spconv_algo
        
        # Always use spconv as backend for now
        set_backend('spconv')
        
        logger.info(f"Environment configured - Backend: spconv, "
                   f"Attention: {self.attention_backend}, "
                   f"Smooth K: {self.smooth_k}, "
                   f"SpConv Algo: {self.spconv_algo}")

# Global config instance
TRELLIS_CONFIG = TrellisConfig()

class IF_TrellisCheckpointLoader:
    """
    Node to manage the loading of the TRELLIS model.
    Follows ComfyUI conventions for model management.
    """
    def __init__(self):
        self.model_manager = None
        # Check for available devices
        self.device = self._get_device()
        
    def _get_device(self):
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define input types with device-specific options."""
        device_options = []
        if torch.cuda.is_available():
            device_options.append("cuda")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_options.append("mps")
        device_options.append("cpu")

        return {
            "required": {
                "model_name": (["TRELLIS-image-large"],),
                "dinov2_model": (["dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg"], {"default": "dinov2_vitl14_reg", "tooltip": "Select the Dinov2 model to use for the image to 3D conversion. Smaller models work but better results with larger models."}),
                "use_fp16": ("BOOLEAN", {"default": True}),
                "attn_backend": (["sage", "xformers", "flash_attn", "sdpa", "naive"], {"default": "sage", "tooltip": "Select the attention backend to use for the image to 3D conversion. Sage is experimental but faster"}),
                "smooth_k": ("BOOLEAN", {"default": True, "tooltip": "Smooth k for sage attention. This is a hyperparameter that controls the smoothness of the attention distribution. It is a boolean value that determines whether to use smooth k or not. Smooth k is a hyperparameter that controls the smoothness of the attention distribution. It is a boolean value that determines whether to use smooth k or not."}),
                "spconv_algo": (["implicit_gemm", "native"], {"default": "implicit_gemm", "tooltip": "Select the spconv algorithm to use for the image to 3D conversion. Implicit gemm is the best but slower. Native is the fastest but less accurate."}),
                "main_device": (device_options, {"default": device_options[0]}),
            },
        }
    
    RETURN_TYPES = ("TRELLIS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "ImpactFrames💥🎞️/Trellis"

    @classmethod
    def _check_backend_availability(cls, backend: str) -> bool:
        """Check if a specific attention backend is available"""
        try:
            if backend == 'sage':
                import sageattention
            elif backend == 'xformers':
                import xformers.ops
            elif backend == 'flash_attn':
                import flash_attn
            elif backend in ['sdpa', 'naive']:
                # These are always available in PyTorch
                pass
            else:
                return False
            return True
        except ImportError:
            return False

    @classmethod
    def _initialize_backend(cls, requested_backend: str = None) -> str:
        """Initialize attention backend with fallback logic"""
        # Priority order for backends
        backend_priority = ['sage', 'flash_attn', 'xformers', 'sdpa']
        
        # If a specific backend is requested, try it first
        if requested_backend:
            if cls._check_backend_availability(requested_backend):
                logger.info(f"Using requested attention backend: {requested_backend}")
                return requested_backend
            else:
                logger.warning(f"Requested backend '{requested_backend}' not available, falling back")
        
        # Try backends in priority order
        for backend in backend_priority:
            if cls._check_backend_availability(backend):
                logger.info(f"Using attention backend: {backend}")
                return backend
        
        # Final fallback to SDPA
        logger.info("All optimized attention backends unavailable, using PyTorch SDPA")
        return 'sdpa'

    def _setup_environment(self):
        """
        Set up environment variables based on the global TRELLIS_CONFIG.
        """
        import os
        from trellis.modules import set_attention_backend
        from trellis.modules.sparse import set_backend
        from trellis.modules.sparse.conv import SPCONV_ALGO

        # Set attention backend
        os.environ['ATTN_BACKEND'] = TRELLIS_CONFIG.attention_backend
        set_attention_backend(TRELLIS_CONFIG.attention_backend)

        # Set smooth k for sage attention
        os.environ['SAGEATTN_SMOOTH_K'] = '1' if TRELLIS_CONFIG.smooth_k else '0'

        # Set spconv algorithm
        os.environ['SPCONV_ALGO'] = TRELLIS_CONFIG.spconv_algo
        
        # Always use spconv as backend for now
        set_backend('spconv')

        logger.info(f"Environment configured - Backend: spconv, "
                    f"Attention: {TRELLIS_CONFIG.attention_backend}, "
                    f"Smooth K: {TRELLIS_CONFIG.smooth_k}, "
                    f"SpConv Algo: {TRELLIS_CONFIG.spconv_algo}")

    def optimize_pipeline(self, pipeline, use_fp16=True, attn_backend='sage'):
        """Apply optimizations to the pipeline if available"""
        if self.device == "cuda":
            try:
                if hasattr(pipeline, 'cuda'):
                    pipeline.cuda()
                    
                if use_fp16:
                    if hasattr(pipeline, 'enable_attention_slicing'):
                        pipeline.enable_attention_slicing()
                    if hasattr(pipeline, 'half'):
                        pipeline.half()
                    
                # Only enable xformers if using xformers backend
                if attn_backend == 'xformers' and hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                    pipeline.enable_xformers_memory_efficient_attention()
                    
            except Exception as e:
                logger.warning(f"Some optimizations failed: {str(e)}")
                
        return pipeline

    def load_model(self, model_name, dinov2_model="dinov2_vitg14", attn_backend="sage", use_fp16=True,
                  smooth_k=True, spconv_algo="implicit_gemm", main_device="cuda"):
        """Load and configure the TRELLIS model."""
        try:
            # Update global config
            TRELLIS_CONFIG.attention_backend = attn_backend
            TRELLIS_CONFIG.spconv_algo = spconv_algo
            TRELLIS_CONFIG.smooth_k = smooth_k
            TRELLIS_CONFIG.device = main_device
            TRELLIS_CONFIG.use_fp16 = use_fp16
            TRELLIS_CONFIG.set("dinov2_model", dinov2_model)

            # Set up environment
            self._setup_environment()

            # Configure attention backend
            set_attention_backend(attn_backend)
            if attn_backend == 'sage':
                enable_sage_attention()
            else:
                disable_sage_attention()

            # Get model path
            model_path = folder_paths.get_full_path("checkpoints", model_name)
            if model_path is None:
                model_path = os.path.join(folder_paths.models_dir, "checkpoints", model_name)

            # Create pipeline with specified dinov2 model
            pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path, dinov2_model=dinov2_model)
            
            # Configure pipeline after loading
            pipeline._device = torch.device(main_device)
            pipeline.attention_backend = attn_backend
            
            # Store configuration in pipeline
            pipeline.config = {
                'device': main_device,
                'use_fp16': use_fp16,
                'attention_backend': attn_backend,
                'dinov2_model': dinov2_model,
                'spconv_algo': spconv_algo,
                'smooth_k': smooth_k
            }

            # Apply optimizations
            pipeline = self.optimize_pipeline(pipeline, use_fp16, attn_backend)

            return (pipeline,)

        except Exception as e:
            logger.error(f"Error loading TRELLIS model: {str(e)}")
            raise