#!/usr/bin/env python3
"""
Script to run the complete data processing pipeline for SPIRIT.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from data_processing import SolarDataProcessor
from helper import load_config, validate_config, log_experiment_info


def main():
    parser = argparse.ArgumentParser(description='Run SPIRIT data processing pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--base-path', type=str, help='Base save path for data')
    parser.add_argument('--csv-path', type=str, help='Path to main CSV file')
    parser.add_argument('--hf-token', type=str, help='Hugging Face token')
    parser.add_argument('--dataset-name', type=str, help='Hugging Face dataset name')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.start_date:
        config['data']['start_date'] = args.start_date
    if args.end_date:
        config['data']['end_date'] = args.end_date
    if args.base_path:
        config['data']['base_save_path'] = args.base_path
    if args.csv_path:
        config['data']['main_csv_path'] = args.csv_path
    if args.hf_token:
        config['huggingface']['token'] = args.hf_token
    if args.dataset_name:
        config['huggingface']['dataset_name'] = args.dataset_name
    
    # Validate configuration
    validate_config(config)
    
    # Log experiment info
    log_experiment_info(config, "Data Processing")
    
    # Initialize processor
    processor = SolarDataProcessor(config)
    
    try:
        # Run complete pipeline
        df = processor.process_complete_pipeline()
        print(f"\nData processing completed successfully!")
        print(f"Final dataset shape: {df.shape}")
        print(f"Dataset uploaded to: {config['huggingface']['dataset_name']}")
        
    except Exception as e:
        print(f"Error during data processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()