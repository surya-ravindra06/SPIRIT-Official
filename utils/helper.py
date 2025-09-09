import yaml
import os
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, date


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def parse_date(date_string: str) -> date:
    """Parse date string to date object."""
    return datetime.strptime(date_string, "%Y-%m-%d").date()


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters."""
    required_sections = ['data', 'huggingface', 'embeddings', 'training']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate data section
    data_config = config['data']
    required_data_fields = ['start_date', 'end_date', 'base_save_path', 'main_csv_path']
    
    for field in required_data_fields:
        if field not in data_config:
            raise ValueError(f"Missing required data configuration field: {field}")
    
    # Validate date format
    try:
        start_date = parse_date(data_config['start_date'])
        end_date = parse_date(data_config['end_date'])
        
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
    except ValueError as e:
        raise ValueError(f"Invalid date format: {e}")
    
    # Validate paths exist
    if not os.path.exists(data_config['main_csv_path']):
        raise ValueError(f"Main CSV file not found: {data_config['main_csv_path']}")
    
    return True


def log_experiment_info(config: Dict[str, Any], model_type: str) -> None:
    """Log experiment information."""
    print(f"\n{'='*50}")
    print(f"SPIRIT - {model_type.upper()} EXPERIMENT")
    print(f"{'='*50}")
    print(f"Date Range: {config['data']['start_date']} to {config['data']['end_date']}")
    print(f"Embedding Model: {config['embeddings']['model_name']}")
    print(f"Study Name: {config['training'].get('study_name', 'Not specified')}")
    print(f"{'='*50}\n")


def print_metrics(metrics: Dict[str, float], title: str = "Model Performance") -> None:
    """Print metrics in a formatted way."""
    print(f"\n{title}")
    print("-" * len(title))
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print()


def save_results(results: Dict[str, Any], save_path: str) -> None:
    """Save experiment results."""
    import json
    
    # Convert numpy types to native Python types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results_serializable = {k: convert_numpy(v) for k, v in results.items()}
    
    with open(save_path, 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)
    
    print(f"Results saved to {save_path}")


def check_gpu_availability() -> None:
    """Check and print GPU availability."""
    import torch
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        
        print(f"GPU Available: Yes")
        print(f"GPU Count: {gpu_count}")
        print(f"Current GPU: {gpu_name}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("GPU Available: No - Using CPU")


def estimate_processing_time(num_images: int, processing_type: str = "embeddings") -> str:
    """Estimate processing time based on number of images."""
    # Rough estimates based on typical performance
    time_per_image = {
        "embeddings": 0.1,  # seconds per image
        "download": 0.5,    # seconds per image
        "processing": 0.05  # seconds per image
    }
    
    total_seconds = num_images * time_per_image.get(processing_type, 0.1)
    
    if total_seconds < 60:
        return f"{total_seconds:.0f} seconds"
    elif total_seconds < 3600:
        return f"{total_seconds/60:.1f} minutes"
    else:
        return f"{total_seconds/3600:.1f} hours"