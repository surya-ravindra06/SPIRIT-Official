#!/usr/bin/env python3
"""
Script to generate image embeddings for SPIRIT.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from embeddings import EmbeddingGenerator
from helper import load_config, log_experiment_info, check_gpu_availability, estimate_processing_time


def main():
    parser = argparse.ArgumentParser(description='Generate image embeddings for SPIRIT')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset-name', type=str, help='Hugging Face dataset name')
    parser.add_argument('--model-name', type=str, help='Embedding model name')
    parser.add_argument('--output-file', type=str, help='Output embeddings file path')
    parser.add_argument('--start-index', type=int, default=0,
                       help='Start index for processing (useful for resuming)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        dataset_name = config['huggingface']['dataset_name']
    
    if args.model_name:
        config['embeddings']['model_name'] = args.model_name
    if args.output_file:
        config['embeddings']['output_file'] = args.output_file
    
    # Log experiment info
    log_experiment_info(config, "Embedding Generation")
    
    # Check GPU availability
    check_gpu_availability()
    
    # Initialize embedding generator
    try:
        generator = EmbeddingGenerator(config)
        
        # Estimate processing time
        from datasets import load_dataset
        dataset = load_dataset(dataset_name, split='train')
        num_images = len(dataset) - args.start_index
        estimated_time = estimate_processing_time(num_images, "embeddings")
        print(f"Estimated processing time: {estimated_time}")
        
        # Generate embeddings
        generator.process_dataset(dataset_name, start_index=args.start_index)
        
        print(f"\nEmbedding generation completed successfully!")
        print(f"Embeddings saved to: {config['embeddings']['output_file']}")
        
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()