"""Model management utilities for downloading and loading pretrained models."""

import torch
from pathlib import Path
import requests
from tqdm import tqdm


class ModelManager:
    """Manager for downloading and caching pretrained models."""
    
    def __init__(self, cache_dir=None):
        """Initialize model manager.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parent.parent / "pretrained_models"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model URLs (using public pretrained weights)
        self.model_urls = {
            'rrdb_esrgan': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'zero_dce': None,  # Will train from scratch
        }
    
    def download_model(self, model_name, force=False):
        """Download pretrained model.
        
        Args:
            model_name: Name of model to download
            force: Force redownload even if cached
            
        Returns:
            Path to downloaded model file
        """
        if model_name not in self.model_urls:
            raise ValueError(f"Unknown model: {model_name}")
        
        url = self.model_urls[model_name]
        if url is None:
            print(f"No pretrained weights available for {model_name}")
            return None
        
        # Check cache
        cache_path = self.cache_dir / f"{model_name}.pth"
        if cache_path.exists() and not force:
            print(f"Using cached model: {cache_path}")
            return cache_path
        
        print(f"Downloading {model_name} from {url}...")
        
        try:
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(cache_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"Downloaded to {cache_path}")
            return cache_path
            
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
            if cache_path.exists():
                cache_path.unlink()
            return None
    
    def load_pretrained_weights(self, model, model_name):
        """Load pretrained weights into model.

        Args:
            model: PyTorch model instance
            model_name: Name of model weights to load
        """
        checkpoint_path = self.download_model(model_name)

        if checkpoint_path is None:
            print(f"No pretrained weights loaded for {model_name}")
            return False

        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')

            # Handle checkpoints with nested state_dict
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights for {model_name}")
            return True

        except Exception as e:
            print(f"Failed to load weights for {model_name}: {e}")
            return False
