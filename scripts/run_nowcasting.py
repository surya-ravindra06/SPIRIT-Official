#!/usr/bin/env python3
"""
Script to train the nowcasting model for SPIRIT.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from nowcasting import NowcastingModel
from helper import load_config, log_experiment_info, check_gpu_availability, print_metrics, save_results


def main():
    parser = argparse.ArgumentParser(description='Train SPIRIT nowcasting model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset-name', type=str, help='Hugging Face dataset name')
    parser.add_argument('--embeddings-file', type=str, help='Path to embeddings file')
    parser.add_argument('--model-save-path', type=str, help='Path to save trained model')
    parser.add_argument('--wandb-key', type=str, help='Weights & Biases API key')
    parser.add_argument('--n-trials', type=int, help='Number of optimization trials')
    parser.add_argument('--model-type', type=str, 
                       choices=['xgboost', 'randomforest', 'mlp'],
                       help='Type of regression model to use')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        dataset_name = config['huggingface']['dataset_name']
    
    if args.embeddings_file:
        embeddings_file = args.embeddings_file
    else:
        embeddings_file = config['embeddings']['output_file']
    
    if args.model_save_path:
        model_save_path = args.model_save_path
    else:
        model_save_path = config['training']['nowcast_model_path']
    
    if args.wandb_key:
        config['training']['wandb_key'] = args.wandb_key
    if args.n_trials:
        config['nowcasting']['n_trials'] = args.n_trials
    if args.model_type:
        config['nowcasting']['model_type'] = args.model_type
        config['training']['wandb_key'] = args.wandb_key
    if args.n_trials:
        config['nowcasting']['n_trials'] = args.n_trials
    
    # Log experiment info
    log_experiment_info(config, "Nowcasting Training")
    
    # Check GPU availability (though XGBoost doesn't use GPU by default)
    check_gpu_availability()
    
    # Initialize nowcasting model
    try:
        model_trainer = NowcastingModel(config)
        
        print(f"Dataset: {dataset_name}")
        print(f"Embeddings: {embeddings_file}")
        print(f"Model save path: {model_save_path}")
        print(f"Model type: {config['nowcasting'].get('model_type', 'xgboost').upper()}")
        print(f"Number of trials: {config['nowcasting'].get('n_trials', 100)}")
        
        # Train model
        final_model, best_params = model_trainer.train_final_model(
            dataset_name, embeddings_file, model_save_path
        )
        
        print(f"\nNowcasting model training completed successfully!")
        print_metrics(best_params, "Best Hyperparameters")
        
        # Save experiment results
        results = {
            'model_type': 'nowcasting',
            'regression_model_type': config['nowcasting'].get('model_type', 'xgboost'),
            'dataset_name': dataset_name,
            'embeddings_file': embeddings_file,
            'model_save_path': model_save_path,
            'best_params': best_params,
            'embedding_model': config['embeddings']['model_name']
        }
        
        results_path = model_save_path.replace('.json', '_results.json')
        save_results(results, results_path)
        
    except Exception as e:
        print(f"Error during nowcasting training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()