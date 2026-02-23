"""Model management utilities for downloading and loading pretrained models."""

import torch
import os
from pathlib import Path
import gdown
import requests
from tqdm import tqdm


class ModelManager:
    """Manager for downloading and caching pretrained models."""
    
    def __init__(self, cache_dir='/app/backend/pretrained_models'):
        """Initialize model manager.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
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
            model: PyTorch model
            model_name: Name of pretrained model
            
        Returns:
            Model with loaded weights
        """
        checkpoint_path = self.download_model(model_name)
        
        if checkpoint_path is None:
            print(f"Cannot load pretrained weights for {model_name}")
            return model
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'params' in checkpoint:
                state_dict = checkpoint['params']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Try to load, handling key mismatches
            try:
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained weights for {model_name}")
            except RuntimeError as e:
                print(f"Warning: Could not load all weights: {e}")
                # Load what we can
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() 
                                  if k in model_dict and v.shape == model_dict[k].shape}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
        
        return model
    
    def get_checkpoint_path(self, model_name, epoch=None):
        """Get path for saving checkpoint.
        
        Args:
            model_name: Name of model
            epoch: Epoch number (None for 'latest')
            
        Returns:
            Path for checkpoint
        """
        checkpoint_dir = Path('/app/backend/outputs/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if epoch is None:
            filename = f"{model_name}_latest.pth"
        else:
            filename = f"{model_name}_epoch{epoch:04d}.pth"
        
        return checkpoint_dir / filename
    
    def save_checkpoint(self, model, optimizer, epoch, model_name, 
                       loss=None, metrics=None):
        """Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            model_name: Name of model
            loss: Current loss value
            metrics: Dictionary of metrics
        """
        checkpoint_path = self.get_checkpoint_path(model_name, epoch)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        if loss is not None:
            checkpoint['loss'] = loss
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as 'latest'
        latest_path = self.get_checkpoint_path(model_name, None)
        torch.save(checkpoint, latest_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, model, model_name, optimizer=None, epoch=None):
        """Load model checkpoint.
        
        Args:
            model: PyTorch model
            model_name: Name of model
            optimizer: Optimizer (optional)
            epoch: Specific epoch to load (None for latest)
            
        Returns:
            Tuple of (model, optimizer, epoch, metrics)
        """
        checkpoint_path = self.get_checkpoint_path(model_name, epoch)
        
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return model, optimizer, 0, None
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            epoch = checkpoint.get('epoch', 0)
            metrics = checkpoint.get('metrics', None)
            
            print(f"Loaded checkpoint from epoch {epoch}")
            
            return model, optimizer, epoch, metrics
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return model, optimizer, 0, None
