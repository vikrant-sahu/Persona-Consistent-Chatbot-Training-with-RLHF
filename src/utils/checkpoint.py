import torch
import os
import shutil
from typing import Dict, Any

class CheckpointManager:
    """Manage model checkpoints during training"""
    
    def __init__(self, base_dir: str = "models/checkpoints", max_checkpoints: int = 3):
        self.base_dir = base_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(base_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, step: int, path: str = None):
        """Save training checkpoint"""
        if path is None:
            path = f"{self.base_dir}/checkpoint-{step}"
        
        os.makedirs(path, exist_ok=True)
        
        # Save model and optimizer state
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(path)
        else:
            torch.save(model.state_dict(), f"{path}/model.pt")
        
        if optimizer:
            torch.save(optimizer.state_dict(), f"{path}/optimizer.pt")
        
        # Save training metadata
        metadata = {
            'step': step,
            'timestamp': torch.tensor(torch.timestamp())
        }
        torch.save(metadata, f"{path}/metadata.pt")
        
        print(f"Checkpoint saved: {path}")
        
        # Clean up old checkpoints
        self.cleanup_old_checkpoints()
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load training checkpoint"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = {}
        
        # Load model weights
        model_path = f"{path}/pytorch_model.bin"
        if os.path.exists(model_path):
            checkpoint['model_state_dict'] = torch.load(model_path)
        elif os.path.exists(f"{path}/model.pt"):
            checkpoint['model_state_dict'] = torch.load(f"{path}/model.pt")
        
        # Load optimizer state
        optimizer_path = f"{path}/optimizer.pt"
        if os.path.exists(optimizer_path):
            checkpoint['optimizer_state_dict'] = torch.load(optimizer_path)
        
        # Load metadata
        metadata_path = f"{path}/metadata.pt"
        if os.path.exists(metadata_path):
            checkpoint['metadata'] = torch.load(metadata_path)
        
        return checkpoint
    
    def save_best_model(self, model, metric: float, path: str):
        """Save best model based on metric"""
        best_metric_path = f"{self.base_dir}/best_metric.txt"
        current_best = float('-inf')
        
        if os.path.exists(best_metric_path):
            with open(best_metric_path, 'r') as f:
                current_best = float(f.read().strip())
        
        if metric > current_best:
            # Save new best model
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(path)
            else:
                torch.save(model.state_dict(), f"{path}/model.pt")
            
            # Update best metric
            with open(best_metric_path, 'w') as f:
                f.write(str(metric))
            
            print(f"New best model saved with metric: {metric:.4f}")
    
    def cleanup_old_checkpoints(self, keep_last_n: int = None):
        """Remove old checkpoints, keeping only the most recent ones"""
        if keep_last_n is None:
            keep_last_n = self.max_checkpoints
        
        checkpoints = []
        for item in os.listdir(self.base_dir):
            if item.startswith('checkpoint-'):
                try:
                    step = int(item.split('-')[1])
                    checkpoints.append((step, item))
                except ValueError:
                    continue
        
        # Sort by step and keep only the most recent
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        for step, checkpoint_dir in checkpoints[keep_last_n:]:
            checkpoint_path = os.path.join(self.base_dir, checkpoint_dir)
            shutil.rmtree(checkpoint_path)
            print(f"Removed old checkpoint: {checkpoint_path}")