# SPIRIT: Short-term Prediction of solar IRradIance for zero-shot Transfer learning using Foundation Models

[![arXiv](https://img.shields.io/badge/arXiv-2502.10307-b31b1b.svg)](https://arxiv.org/pdf/2502.10307)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

## Overview

SPIRIT is a novel approach leveraging foundation models for solar irradiance forecasting, enabling zero-shot transfer learning to new locations without historical data. Our method outperforms state-of-the-art models by up to **70%** in zero-shot scenarios.

The system combines pre-trained Vision Transformers (ViT) with physics-informed features to create a comprehensive solar irradiance prediction framework that works across different geographical locations without requiring local training data.

## Repository Structure

```
SPIRIT/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── config/
│   └── config.yaml               # Configuration parameters
├── src/                          # Core implementation modules
│   ├── README.md                 # Source code documentation
│   ├── __init__.py
│   ├── data_processing.py        # Data pipeline and preprocessing
│   ├── embeddings.py             # Vision Transformer embeddings
│   ├── nowcasting.py             # Modular regression models
│   └── forecasting.py            # Temporal forecasting models
├── scripts/                      # Execution scripts
│   ├── README.md                 # Scripts documentation
│   ├── run_data_processing.py    # Data processing pipeline
│   ├── run_embeddings.py         # Embedding generation
│   ├── run_nowcasting.py         # Nowcasting model training
│   └── run_forecasting.py        # Forecasting model training
└── utils/                        # Utility functions
    ├── __init__.py
    └── helper.py                 # Configuration and helper functions
```

## Key Features

- **Zero-shot Transfer Learning**: Deploy at new solar sites without historical data
- **Vision Transformer Integration**: Extract rich features from sky images
- **Modular Regression Framework**: Choose from XGBoost, Random Forest, or MLP
- **Physics-Informed Features**: Solar geometry and clear-sky modeling
- **End-to-End Pipeline**: From raw data to trained models
- **Hyperparameter Optimization**: Automated tuning with Optuna

## Model Architecture

### Real-Time Solar Irradiance Estimation (Nowcasting)

The nowcasting system provides immediate predictions of solar irradiance using a sophisticated fusion approach:

#### Input Processing
- **Sky Images**: Processed through pre-trained Vision Transformers (ViT-huge-patch14-224) to extract high-dimensional visual features
- **Physics Features**: Solar geometry calculations including:
  - Zenith and azimuth angles with trigonometric transformations
  - Panel tilt and orientation parameters
  - Angle of incidence (AOI) calculations
  - Clear-sky Global Horizontal Irradiance (GHI) estimates
  - Tilted irradiance projections

#### Model Options
The system supports three regression algorithms, each optimized for different use cases:

1. **XGBoost (default)**
   - **Best for**: Complex non-linear relationships, fast training and inference
   - **Architecture**: Gradient boosting with tree ensembles
   - **Optimization**: 8 hyperparameters including depth, learning rate, regularization
   - **File format**: `.json`

2. **Random Forest**
   - **Best for**: Robust predictions, high interpretability, stable performance
   - **Architecture**: Ensemble of decision trees with bootstrap sampling
   - **Optimization**: 6 hyperparameters including tree count, splitting criteria
   - **File format**: `.pkl`

3. **Multi-Layer Perceptron (MLP)**
   - **Best for**: Complex patterns, deep learning approach, non-linear mapping
   - **Architecture**: Configurable neural network with 1-3 hidden layers
   - **Optimization**: 7 hyperparameters including layer sizes, activation functions
   - **File format**: `.pkl`

#### Feature Engineering
The system combines 384-dimensional ViT embeddings with 17 physics-informed features:
- Raw physics features (7): Clear-sky GHI, angles, tilt parameters, AOI, tilted irradiance
- Trigonometric features (10): Sine and cosine transformations of all angular features

### Short-Term Solar Forecasting (1–4 Hours Ahead)

The forecasting system predicts future solar irradiance based on temporal dynamics:

#### Architecture
- **Input**: Rolling sequence of sky image embeddings over past 6 hours
- **Encoder**: Transformer architecture with attention mechanisms and positional encoding
- **Decoder**: Residual MLP stack for fine-grained temporal dependencies
- **Output**: Forecasted GHI values across future 15-minute intervals (1-4 hour horizon)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/SPIRIT.git
cd SPIRIT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The system uses a YAML configuration file for all parameters. Copy and modify the template:

```bash
cp config/config.yaml config/my_config.yaml
```

### Key Configuration Sections

#### Data Processing
```yaml
data:
  start_date: "2023-01-01"          # Data collection start
  end_date: "2023-12-31"            # Data collection end
  base_save_path: "/path/to/data"   # Base directory for processed data
  main_csv_path: "/path/to/main.csv" # Main dataset file
  raw_images_dir: "raw_images"      # Raw satellite images
  processed_images_dir: "processed_images" # Processed images
  final_dataset_name: "processed_dataset.csv" # Final dataset filename
```

#### Model Configuration
```yaml
nowcasting:
  study_name: "spirit_nowcasting"   # Optuna study name
  n_trials: 100                     # Hyperparameter optimization trials
  model_type: "xgboost"            # Options: "xgboost", "randomforest", "mlp"

forecasting:
  study_name: "spirit_forecasting"  # Optuna study name
  n_trials: 50                      # Hyperparameter optimization trials
  num_past_frames: 6                # Input sequence length (hours)
  num_future_frames: 24             # Output sequence length (15-min intervals)
  batch_size: 32                    # Training batch size
  num_epochs: 1000                  # Maximum training epochs
```

#### External Services
```yaml
# Hugging Face for dataset hosting
huggingface:
  token: "your_hf_token_here"
  dataset_name: "your-username/dataset-name"  # Use your own HF dataset with same column structure

# Weights & Biases for experiment tracking
training:
  wandb_key: "your_wandb_key_here"
  nowcast_model_path: "/path/to/nowcast_model.json"
  forecast_model_path: "/path/to/forecast_model.pth"
```

> **💡 Using Your Own Dataset**: If you have your own dataset hosted on Hugging Face with the same column structure (sky images, meteorological data, GHI values), simply update the `dataset_name` field above to point to your dataset. The system will automatically work with any dataset that follows the same schema.

## Usage

### Basic Pipeline

#### 1. Data Processing
```bash
# Process raw satellite images and meteorological data
python scripts/run_data_processing.py --config config/my_config.yaml

# With specific date range
python scripts/run_data_processing.py 
    --config config/my_config.yaml 
    --start-date 2023-01-01 
    --end-date 2023-12-31
```

#### 2. Generate Embeddings
```bash
# Extract ViT features from processed images
python scripts/run_embeddings.py --config config/my_config.yaml

# With custom batch size
python scripts/run_embeddings.py 
    --config config/my_config.yaml 
    --batch-size 16
```

#### 3. Train Nowcasting Models
```bash
# Train with XGBoost (default)
python scripts/run_nowcasting.py --config config/my_config.yaml

# Train with Random Forest
python scripts/run_nowcasting.py 
    --config config/my_config.yaml 
    --model-type randomforest

# Train with MLP
python scripts/run_nowcasting.py 
    --config config/my_config.yaml 
    --model-type mlp

# Full parameter specification
python scripts/run_nowcasting.py 
    --config config/my_config.yaml 
    --model-type randomforest 
    --n-trials 150 
    --dataset-name your-dataset 
    --embeddings-file path/to/embeddings.json 
    --model-save-path models/rf_model.pkl 
    --wandb-key your_wandb_key
```

#### 4. Train Forecasting Model
```bash
# Train temporal forecasting model
python scripts/run_forecasting.py --config config/my_config.yaml

# With custom parameters
python scripts/run_forecasting.py 
    --config config/my_config.yaml 
    --num-epochs 500 
    --batch-size 16
```

## Modular Nowcasting System

The nowcasting system supports three regression models that can be easily switched:

### XGBoost (Default)
```bash
python scripts/run_nowcasting.py --model-type xgboost
```
- Gradient boosting with excellent out-of-the-box performance
- Fast training and robust to hyperparameters

### Random Forest
```bash
python scripts/run_nowcasting.py --model-type randomforest
```
- Ensemble method with good interpretability
- Provides feature importance rankings

### Multi-Layer Perceptron (MLP)
```bash
python scripts/run_nowcasting.py --model-type mlp
```
- Neural network for complex non-linear relationships
- Flexible architecture optimization

All models use the same input features (384-dimensional ViT embeddings + 17 physics features) and are optimized using Optuna for hyperparameter tuning.

## Hyperparameter Optimization

The system uses Optuna for automated hyperparameter optimization:

- **XGBoost**: Optimizes tree depth, learning rate, regularization
- **Random Forest**: Optimizes number of trees, depth, sampling
- **MLP**: Optimizes architecture, learning rate, regularization

Each model is automatically tuned for your specific dataset and requirements.

## Model Training Pipeline

### Data Flow Architecture
```
Raw Satellite Images → Image Processing → ViT Embeddings
        ↓                     ↓               ↓
Meteorological Data → Feature Engineering → Physics Features
        ↓                     ↓               ↓
    Combined Dataset → Normalization → Train/Validation Split
        ↓                     ↓               ↓
Hyperparameter Optimization → Model Training → Model Evaluation
        ↓                     ↓               ↓
    Best Model → Model Saving → Performance Metrics
```

### Training Process Details

#### 1. Data Loading and Preprocessing
```python
def load_data(self, dataset_name: str, embeddings_file: str):
    """
    Load dataset and embeddings for training.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        embeddings_file: Path to pre-computed ViT embeddings
        
    Returns:
        embeddings: (N, 384) ViT feature vectors
        auxiliary_features: (N, 17) physics-informed features
        targets: (N, 1) Global Horizontal Irradiance values
    """
```

The function performs:
- Dataset loading from HuggingFace
- Embedding file parsing (JSON lines format)
- Physics feature calculation with trigonometric transformations
- Target variable extraction and validation

#### 2. Feature Normalization
```python
def normalize_features(self, X: np.ndarray, fit: bool = True):
    """
    Normalize features using z-score standardization.
    
    Args:
        X: Input features (N, D)
        fit: Whether to compute normalization parameters
        
    Returns:
        X_normalized: Standardized features
    """
```

Normalization ensures:
- Zero mean and unit variance for all features
- Consistent scaling across different feature types
- Prevention of numerical instability during training

#### 3. Hyperparameter Optimization Loop
```python
def objective(self, trial: optuna.Trial, X_train, y_train):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial instance
        X_train: Training features
        y_train: Training targets
        
    Returns:
        validation_score: nMAPE on validation set
    """
```

The optimization process:
- Suggests hyperparameters based on model-specific search spaces
- Trains model with train/validation split (80/20)
- Evaluates performance using multiple metrics
- Logs results to Weights & Biases
- Returns validation nMAPE for optimization

#### 4. Final Model Training
```python
def train_final_model(self, dataset_name, embeddings_file, model_save_path):
    """
    Train final model with optimized hyperparameters.
    
    This method performs the complete training pipeline:
    1. Load and preprocess data
    2. Optimize hyperparameters
    3. Train final model on full dataset
    4. Save model and normalization parameters
    """
```

## Inference and Deployment

Once trained, models can be used for real-time predictions:

```python
from src.nowcasting import NowcastingModel
from utils.helper import load_config

# Load configuration and model
config = load_config('config/config.yaml')
model = NowcastingModel(config)

# Load trained model
model.load_model('path/to/saved/model.pkl')

# Make predictions
predictions = model.predict(new_data)
```

## Performance Analysis

The system provides comprehensive evaluation metrics:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error  
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error
- **sMAPE**: Symmetric Mean Absolute Percentage Error

## Citation

If you use SPIRIT in your research, please cite:

```bibtex
@article{spirit2025,
  title={SPIRIT: Short-term Prediction of solar IRradIance for zero-shot Transfer learning using Foundation Models},
  author={Your Name and Others},
  journal={arXiv preprint arXiv:2502.10307},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
