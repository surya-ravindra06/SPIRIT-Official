# SPIRIT Scripts Documentation

This directory contains executable scripts for running the complete SPIRIT pipeline. Each script corresponds to a specific stage of the solar irradiance prediction workflow.

## Script Overview

```
scripts/
├── run_data_processing.py      # Stage 1: Data collection and preprocessing
├── run_embeddings.py          # Stage 2: Vision Transformer feature extraction
├── run_nowcasting.py          # Stage 3: Real-time irradiance prediction models
├── run_forecasting.py         # Stage 4: Temporal forecasting models
```

## Execution Scripts

### 1. `run_data_processing.py` - Data Pipeline Orchestration

**Purpose**: Orchestrates the complete data collection, processing, and preparation pipeline for training.

#### Command Line Interface
```bash
# Basic usage
python scripts/run_data_processing.py --config config/config.yaml

# Full parameter specification
python scripts/run_data_processing.py \
    --config config/my_config.yaml \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --base-path /path/to/data \
    --main-csv /path/to/main.csv \
    --dataset-name username/solar-dataset \
    --hf-token your_huggingface_token
```

#### Parameters
| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `--config` | str | Path to YAML configuration file | Yes |
| `--start-date` | str | Data collection start date (YYYY-MM-DD) | No* |
| `--end-date` | str | Data collection end date (YYYY-MM-DD) | No* |
| `--base-path` | str | Base directory for processed data | No* |
| `--main-csv` | str | Path to main dataset CSV file | No* |
| `--dataset-name` | str | HuggingFace dataset identifier | No* |
| `--hf-token` | str | HuggingFace API token | No* |

*Defaults to values in configuration file

#### Implementation Details
```python
def main():
    """
    Main execution function for data processing pipeline.
    
    Pipeline Stages:
    1. Argument parsing and configuration loading
    2. Configuration validation and path setup
    3. SolarDataProcessor initialization
    4. Complete data processing pipeline execution
    5. Dataset upload to HuggingFace Hub
    6. Results logging and cleanup
    """
    
    # Configuration management
    config = load_config(args.config)
    validate_config(config)
    
    # Override config with CLI arguments
    if args.start_date:
        config['data']['start_date'] = args.start_date
    # ... similar for other parameters
    
    # Execute pipeline
    processor = SolarDataProcessor(config)
    final_dataset = processor.process_complete_pipeline()
    
    # Upload and log results
    processor.upload_to_huggingface(final_dataset, dataset_name)
    log_experiment_info(config, "Data Processing")
```

#### Output
- Processed dataset uploaded to HuggingFace Hub
- Local files: processed images, feature CSVs, logs
- Performance metrics and processing statistics

#### Example Usage
```bash
# Process one year of data
python scripts/run_data_processing.py \
    --config config/nrel_site_config.yaml \
    --start-date 2023-01-01 \
    --end-date 2023-12-31

# Quick processing for development
python scripts/run_data_processing.py \
    --config config/dev_config.yaml \
    --start-date 2023-06-01 \
    --end-date 2023-06-07
```

---

### 2. `run_embeddings.py` - Feature Extraction Pipeline

**Purpose**: Extracts high-dimensional features from processed sky images using pre-trained Vision Transformers.

#### Command Line Interface
```bash
# Basic usage
python scripts/run_embeddings.py --config config/config.yaml

# With custom parameters
python scripts/run_embeddings.py \
    --config config/my_config.yaml \
    --dataset-name username/solar-dataset \
    --output-file data/embeddings.json \
    --model-name google/vit-huge-patch14-224-in21k \
    --batch-size 16 \
    --device cuda
```

#### Parameters
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--config` | str | Configuration file path | Required |
| `--dataset-name` | str | HuggingFace dataset to process | From config |
| `--output-file` | str | Path for embedding output | From config |
| `--model-name` | str | ViT model identifier | From config |
| `--batch-size` | int | Processing batch size | 32 |
| `--device` | str | Computing device (cuda/cpu) | Auto-detect |
| `--num-workers` | int | Data loading workers | 4 |

#### Implementation Details
```python
def main():
    """
    Embedding generation pipeline execution.
    
    Process:
    1. Configuration loading and GPU setup
    2. Dataset loading from HuggingFace Hub
    3. ViT model initialization and device placement
    4. Batch processing with progress tracking
    5. Embedding extraction and normalization
    6. Output serialization in JSON Lines format
    7. Validation and quality checks
    """
    
    # Hardware optimization
    check_gpu_availability()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Model setup
    generator = EmbeddingGenerator(config)
    generator.model.to(device)
    
    # Process dataset
    generator.generate_embeddings(
        dataset_name=dataset_name,
        output_file=output_file
    )
    
    # Validation
    validate_embeddings(output_file)
```

#### Performance Optimization
```python
# Memory-efficient processing
def process_large_dataset(generator, dataset, batch_size=16):
    """Process large datasets with memory management."""
    
    for batch_idx, batch in enumerate(dataset.iter(batch_size)):
        embeddings = generator.process_image_batch(batch['images'])
        
        # Save incrementally to avoid memory issues
        save_batch_embeddings(embeddings, batch_idx)
        
        # Periodic cleanup
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
```

#### Output Format
```json
{"image_id": "2023-01-01T12:00:00", "embedding": [0.123, -0.456, ...]}
{"image_id": "2023-01-01T12:15:00", "embedding": [0.789, -0.012, ...]}
```

#### Example Usage
```bash
# High-quality embeddings with large batch
python scripts/run_embeddings.py \
    --config config/production.yaml \
    --batch-size 32 \
    --device cuda

# Memory-constrained environment
python scripts/run_embeddings.py \
    --config config/config.yaml \
    --batch-size 8 \
    --device cpu \
    --num-workers 2
```

---

### 3. `run_nowcasting.py` - Modular Regression Training

**Purpose**: Train and optimize regression models for real-time solar irradiance prediction with support for multiple algorithms.

#### Command Line Interface
```bash
# Basic usage with XGBoost (default)
python scripts/run_nowcasting.py --config config/config.yaml

# Train Random Forest model
python scripts/run_nowcasting.py \
    --config config/config.yaml \
    --model-type randomforest

# Full parameter specification
python scripts/run_nowcasting.py \
    --config config/my_config.yaml \
    --model-type mlp \
    --dataset-name username/solar-dataset \
    --embeddings-file data/embeddings.json \
    --model-save-path models/mlp_nowcast.pkl \
    --n-trials 200 \
    --wandb-key your_wandb_api_key
```

#### Parameters
| Parameter | Type | Choices | Description | Required |
|-----------|------|---------|-------------|----------|
| `--config` | str | - | Configuration file path | Yes |
| `--model-type` | str | `xgboost`, `randomforest`, `mlp` | Regression algorithm | No* |
| `--dataset-name` | str | - | HuggingFace dataset identifier | No* |
| `--embeddings-file` | str | - | Path to ViT embeddings | No* |
| `--model-save-path` | str | - | Output path for trained model | No* |
| `--n-trials` | int | 1-1000 | Optuna optimization trials | No* |
| `--wandb-key` | str | - | Weights & Biases API key | No* |

*Defaults to configuration file values

#### Model-Specific Behavior
```python
def setup_model_specific_config(model_type: str, config: dict) -> dict:
    """Configure model-specific parameters."""
    
    if model_type == 'xgboost':
        # Fast training, good default performance
        config['nowcasting']['n_trials'] = config.get('n_trials', 100)
        expected_time = "5-15 minutes"
        
    elif model_type == 'randomforest':
        # Medium training time, high interpretability
        config['nowcasting']['n_trials'] = config.get('n_trials', 75)
        expected_time = "10-30 minutes"
        
    elif model_type == 'mlp':
        # Longer training, complex patterns
        config['nowcasting']['n_trials'] = config.get('n_trials', 50)
        expected_time = "30-60 minutes"
    
    return config, expected_time
```

#### Implementation Details
```python
def main():
    """
    Complete nowcasting model training pipeline.
    
    Workflow:
    1. Argument parsing and configuration management
    2. Model type validation and setup
    3. Data loading and preprocessing
    4. Hyperparameter optimization with Optuna
    5. Final model training on full dataset
    6. Model evaluation and validation
    7. Model and metadata persistence
    8. Results logging and experiment tracking
    """
    
    # Configuration setup
    config = load_config(args.config)
    model_type = args.model_type or config['nowcasting'].get('model_type', 'xgboost')
    
    # Model initialization
    model_trainer = NowcastingModel(config)
    print(f"Training {model_type.upper()} model...")
    
    # Training execution
    final_model, best_params = model_trainer.train_final_model(
        dataset_name, embeddings_file, model_save_path
    )
    
    # Results management
    save_training_results(final_model, best_params, model_type)
    log_to_wandb(best_params, model_type)
```

#### Hyperparameter Optimization Details
```python
def optimize_model_hyperparameters(model_trainer, X_train, y_train, model_type):
    """
    Model-specific optimization strategies.
    """
    
    optimization_configs = {
        'xgboost': {
            'pruner': optuna.pruners.MedianPruner(n_startup_trials=5),
            'sampler': optuna.samplers.TPESampler(seed=42),
            'direction': 'minimize'
        },
        'randomforest': {
            'pruner': optuna.pruners.MedianPruner(n_startup_trials=10),
            'sampler': optuna.samplers.RandomSampler(seed=42),
            'direction': 'minimize'
        },
        'mlp': {
            'pruner': optuna.pruners.MedianPruner(n_startup_trials=15),
            'sampler': optuna.samplers.TPESampler(seed=42),
            'direction': 'minimize'
        }
    }
    
    config = optimization_configs[model_type]
    # Create study with model-specific configuration
    # ... optimization logic
```

#### Output Files
```
models/
├── {model_type}_model.{pkl|json}     # Trained model
├── {model_type}_normalization.json   # Feature normalization params
├── {model_type}_results.json         # Training results and metadata
└── optuna_study_{model_type}.db      # Optimization history
```

#### Example Usage
```bash
# Compare all models
for model in xgboost randomforest mlp; do
    python scripts/run_nowcasting.py \
        --config config/config.yaml \
        --model-type $model \
        --model-save-path models/${model}_model
done

# Quick development testing
python scripts/run_nowcasting.py \
    --config config/dev_config.yaml \
    --model-type xgboost \
    --n-trials 10
```

---

### 4. `run_forecasting.py` - Temporal Forecasting Training

**Purpose**: Train Transformer-based models for 1-4 hour solar irradiance forecasting using temporal sequences.

#### Command Line Interface
```bash
# Basic usage
python scripts/run_forecasting.py --config config/config.yaml

# With custom parameters
python scripts/run_forecasting.py \
    --config config/my_config.yaml \
    --dataset-name username/solar-dataset \
    --embeddings-file data/embeddings.json \
    --model-save-path models/forecast_model.pth \
    --num-epochs 500 \
    --batch-size 16 \
    --learning-rate 0.001
```

#### Parameters
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--config` | str | Configuration file path | Required |
| `--dataset-name` | str | HuggingFace dataset identifier | From config |
| `--embeddings-file` | str | Path to ViT embeddings | From config |
| `--model-save-path` | str | Output path for trained model | From config |
| `--num-epochs` | int | Maximum training epochs | 1000 |
| `--batch-size` | int | Training batch size | 32 |
| `--learning-rate` | float | Initial learning rate | 0.001 |
| `--num-past-frames` | int | Input sequence length (hours) | 6 |
| `--num-future-frames` | int | Output sequence length (15-min intervals) | 24 |

#### Implementation Details
```python
def main():
    """
    Temporal forecasting model training pipeline.
    
    Architecture:
    1. Temporal data preparation and sequence creation
    2. Train/validation/test split with temporal awareness
    3. Transformer model initialization and configuration
    4. Training loop with early stopping and checkpointing
    5. Model evaluation on multiple forecasting horizons
    6. Model persistence and metadata saving
    """
    
    # Data preparation
    forecaster = ForecastingModel(config)
    temporal_data = forecaster.prepare_temporal_data(embeddings, targets)
    
    # Model training
    model = forecaster.train_model(
        dataset_name=dataset_name,
        embeddings_file=embeddings_file
    )
    
    # Evaluation
    evaluate_forecasting_performance(model, test_data)
```

#### Training Process Details
```python
def train_forecasting_model(model, train_loader, val_loader, config):
    """
    Advanced training with multiple optimizations.
    """
    
    # Optimizer setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-5
    )
    
    # Learning rate scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2
    )
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        # Training phase
        train_loss = train_epoch(model, train_loader, optimizer)
        
        # Validation phase
        val_loss = validate_epoch(model, val_loader)
        
        # Learning rate update
        scheduler.step()
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, epoch, val_loss)
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch}")
            break
```

#### Example Usage
```bash
# Full training run
python scripts/run_forecasting.py \
    --config config/production.yaml \
    --num-epochs 1000 \
    --batch-size 32

# Development/testing
python scripts/run_forecasting.py \
    --config config/dev_config.yaml \
    --num-epochs 50 \
    --batch-size 8
```

---
