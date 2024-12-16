#__init__.py
import os
import sys
import logging
import folder_paths

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ComfyUI-IF_Trellis')

# Add parent directory to Python path to find trellis package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add both current and parent dir to handle different installation scenarios
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add trellis package path
trellis_path = os.path.join(current_dir, "trellis")
if os.path.exists(trellis_path) and trellis_path not in sys.path:
    sys.path.insert(0, trellis_path)
    logger.info(f"Added trellis path to sys.path: {trellis_path}")

# Verify trellis package is importable
try:
    import trellis
    logger.info("Trellis package imported successfully")
except ImportError as e:
    logger.error(f"Failed to import trellis package: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    raise

# Register model paths with ComfyUI
try:
    folder_paths.add_model_folder_path("trellis", os.path.join(folder_paths.models_dir, "trellis"))
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.models_dir, "checkpoints"))
except Exception as e:
    logger.error(f"Error registering model paths: {e}")

# Ensure required directories exist
def setup_directories():
    """Create required directories if they don't exist"""
    required_dirs = [
        os.path.join(folder_paths.models_dir, "checkpoints", "TRELLIS-image-large"),
        os.path.join(folder_paths.models_dir, "checkpoints", "TRELLIS-image-large", "ckpts"),
        os.path.join(current_dir, "web", "js"),
        os.path.join(current_dir, "web", "html")
    ]
    
    for directory in required_dirs:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Directory ensured: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")

    web_dirs = [
        os.path.join(current_dir, "web", "js", "lib"),  # For third-party libraries
        os.path.join(current_dir, "web", "js", "three"), # For Three.js modules
    ]
    for directory in web_dirs:
        os.makedirs(directory, exist_ok=True)

# Initialize module
def init():
    """Initialize the module"""
    try:
        setup_directories()
        logger.info("ComfyUI-IF_Trellis initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing ComfyUI-IF_Trellis: {e}")

init()

# Import node classes only after ensuring trellis is importable
try:
    from .IF_TrellisCheckpointLoader import IF_TrellisCheckpointLoader
    from .IF_Trellis import IF_TrellisImageTo3D
    from .IF_Preview_3D import IF_Preview_3D

    # Register web extensions
    WEB_DIRECTORY = "web"
    NODE_CLASS_MAPPINGS = {
        "IF_TrellisCheckpointLoader": IF_TrellisCheckpointLoader,
        "IF_TrellisImageTo3D": IF_TrellisImageTo3D,
        "IF_Preview_3D": IF_Preview_3D
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "IF_TrellisCheckpointLoader": "Trellis Model Loader 💾",
        "IF_TrellisImageTo3D": "Trellis Image to 3D 🖼️➡️🎲",
        "IF_Preview_3D": "Trellis 3D Preview 🎲🔍"
    }

except Exception as e:
    logger.error(f"Error importing node classes: {e}")
    raise

# Clean up namespace
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']