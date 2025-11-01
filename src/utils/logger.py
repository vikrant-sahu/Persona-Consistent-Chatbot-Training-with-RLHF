import wandb
import logging
from typing import Dict, Any, List
import os

class Logger:
    """Weights & Biases logging utility"""
    
    def __init__(self, project: str, config: Dict, entity: str = None):
        self.project = project
        self.config = config
        self.entity = entity
        self.run = None
        
        # Setup file logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('outputs/logs/training.log'),
                logging.StreamHandler()
            ]
        )
        self.file_logger = logging.getLogger()
    
    def init_run(self, run_name: str = None, tags: List[str] = None):
        """Initialize W&B run"""
        if wandb.run is None:
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=run_name,
                tags=tags,
                config=self.config
            )
        else:
            self.run = wandb.run
    
    def log_metrics(self, metrics: Dict, step: int = None):
        """Log metrics to W&B and file"""
        # Log to W&B
        if self.run:
            wandb.log(metrics, step=step)
        
        # Log to file
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.file_logger.info(f"Step {step}: {metrics_str}")
    
    def log_samples(self, samples: List[str], step: int = None):
        """Log generated samples"""
        if self.run:
            # Create a table of samples
            samples_table = wandb.Table(columns=["Step", "Sample"])
            for i, sample in enumerate(samples):
                samples_table.add_data(step if step else i, sample)
            
            wandb.log({"generated_samples": samples_table}, step=step)
    
    def log_model(self, model_path: str):
        """Log model artifacts"""
        if self.run:
            model_artifact = wandb.Artifact(
                name=f"model-{self.run.id}",
                type="model",
                description="Trained persona-consistent chatbot"
            )
            model_artifact.add_dir(model_path)
            self.run.log_artifact(model_artifact)
    
    def finish(self):
        """Finish W&B run"""
        if self.run:
            wandb.finish()