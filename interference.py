
import numpy as np
import cv2
from pathlib import Path

from models.zero_dce import ZeroDCE
from models.rrdb_generator import RRDBNet
from utils.model_manager import ModelManager


class FaceHallucinationPipeline:
    \"\"\"End-to-end pipeline for low-light face super-resolution.\"\"\"
    
    def __init__(self, device='cpu', use_enhancer=True):
        \"\"\"Initialize pipeline.
        
        Args:
            device: 'cpu' or 'cuda'
            use_enhancer: Whether to use Zero-DCE enhancement
        \"\"\"
        self.device = device
        self.use_enhancer = use_enhancer
        
        # Initialize models
        print(\"Initializing Face Hallucination Pipeline...\")
        
        if use_enhancer:
            self.enhancer = ZeroDCE(device=device, n_iterations=8)
            print(\"✓ Zero-DCE enhancer loaded\")
        else:
            self.enhancer = None
        
        self.generator = RRDBNet(device=device, scale=4)
        print(\"✓ RRDB Generator loaded\")
        
        # Model manager for loading pretrained weights
        self.model_manager = ModelManager()
        
        print(\"Pipeline ready!\")
    
    def load_pretrained_models(self):
        \"\"\"Load pretrained model weights.\"\"\"
        print(\"Loading pretrained weights...\")
        
        # Try to load pretrained RRDB weights
        try:
            checkpoint_path = self.model_manager.download_model('rrdb_esrgan')
            if checkpoint_path:
                self.generator.load_checkpoint(str(checkpoint_path))
        except Exception as e:
            print(f\"Could not load pretrained RRDB: {e}\")
            print(\"Using randomly initialized weights (for demo)\")
        
        # Zero-DCE would need training or pretrained weights
        if self.enhancer:
            checkpoint_path = self.model_manager.get_checkpoint_path('zero_dce', None)
            if checkpoint_path.exists():
                self.enhancer.load_checkpoint(str(checkpoint_path))
            else:
                print(\"Zero-DCE: Using randomly initialized weights (needs training)\")
    
    def preprocess_image(self, image):
        \"\"\"Preprocess input image.
        
        Args:
            image: Input image as numpy array [H, W, 3] in RGB [0, 255]
                   or file path string
            
        Returns:
            Preprocessed tensor [1, 3, H, W] in range [0, 1]
        \"\"\"
        if isinstance(image, (str, Path)):
            # Load from file
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        tensor = torch.from_numpy(image).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        return tensor.to(self.device)
    
    def postprocess_image(self, tensor):
        \"\"\"Convert tensor to displayable image.
        
        Args:
            tensor: Output tensor [1, 3, H, W] or [3, H, W]
            
        Returns:
            Numpy image [H, W, 3] in RGB [0, 255]
        \"\"\"
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        image = tensor.permute(1, 2, 0).cpu().numpy()
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image
    
    def enhance_and_super_resolve(self, input_image):
        \"\"\"Run full pipeline on input image.
        
        Args:
            input_image: Input image (numpy or path)
            
        Returns:
            Dictionary with intermediate and final outputs
        \"\"\"
        # Preprocess
        input_tensor = self.preprocess_image(input_image)
        
        results = {'input': input_tensor}
        
        # Enhancement
        if self.enhancer:
            with torch.no_grad():
                enhanced = self.enhancer.enhance(input_tensor)
            results['enhanced'] = enhanced
        else:
            enhanced = input_tensor
            results['enhanced'] = input_tensor
        
        # Super-resolution
        with torch.no_grad():
            sr_output = self.generator.super_resolve(enhanced)
        results['sr_output'] = sr_output
        
        # Convert to displayable images
        results['input_image'] = self.postprocess_image(input_tensor)
        results['enhanced_image'] = self.postprocess_image(enhanced)
        results['sr_image'] = self.postprocess_image(sr_output)
        
        return results
    
    def process_batch(self, image_paths, output_dir=None):
        \"\"\"Process batch of images.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save results (optional)
            
        Returns:
            List of result dictionaries
        \"\"\"
        results = []
        
        for img_path in image_paths:
            print(f\"Processing: {img_path}\")
            result = self.enhance_and_super_resolve(img_path)
            result['path'] = str(img_path)
            results.append(result)
            
            # Save if output dir provided
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                filename = Path(img_path).stem
                sr_path = output_dir / f\"{filename}_sr.png\"
                cv2.imwrite(
                    str(sr_path),
                    cv2.cvtColor(result['sr_image'], cv2.COLOR_RGB2BGR)
                )
        
        return results
    
    def get_model_info(self):
        \"\"\"Get information about loaded models.
        
        Returns:
            Dictionary with model information
        \"\"\"
        info = {
            'device': self.device,
            'enhancer_enabled': self.enhancer is not None,
            'generator_type': 'RRDB (ESRGAN-style)',
            'upscale_factor': '4x',
        }
        
        return info


def demo_inference():
    \"\"\"Demo inference function for testing.\"\"\"
    print(\"=\"*60)
    print(\"Face Hallucination System - Demo Inference\")
    print(\"=\"*60)
    
    # Initialize pipeline
    pipeline = FaceHallucinationPipeline(device='cpu', use_enhancer=True)
    
    # Try to load pretrained weights
    pipeline.load_pretrained_models()
    
    # Print model info
    info = pipeline.get_model_info()
    print(\"
Model Info:\")
    for key, value in info.items():
        print(f\"  {key}: {value}\")
    
    print(\"
✓ Pipeline is ready for inference!\")
    print(\"
To use:\")
    print(\"  results = pipeline.enhance_and_super_resolve('path/to/image.jpg')\")
    print(\"  sr_image = results['sr_image']  # Enhanced high-res result\")
    
    return pipeline


if __name__ == '__main__':
    pipeline = demo_inference()
"