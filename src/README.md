# SPIRIT Core Modules Documentation

This directory contains the core implementation modules for the SPIRIT solar irradiance prediction system. Each module is designed for a specific component of the pipeline, from data processing to model training.

## Module Overview

```
src/
├── __init__.py              # Package initialization
├── data_processing.py       # Solar data pipeline and preprocessing
├── embeddings.py           # Vision Transformer feature extraction
├── nowcasting.py           # Modular regression models for nowcasting
└── forecasting.py          # Temporal models for forecasting
```

## Module Details

### 1. `data_processing.py` - SolarDataProcessor

**Purpose**: Complete data pipeline for processing raw solar and meteorological data into ML-ready datasets.

#### Key Classes

##### `SolarDataProcessor`
Main class orchestrating the entire data processing pipeline.

```python
class SolarDataProcessor:
    """
    Comprehensive solar data processing pipeline.
    
    Handles data collection, image processing, physics calculations,
    and dataset creation for both nowcasting and forecasting tasks.
    """
    
    def __init__(self, config: Dict[str, Any])
    def process_complete_pipeline(self) -> pd.DataFrame
    def download_nrel_data(self, start_date: str, end_date: str) -> pd.DataFrame
    def process_images(self, df: pd.DataFrame) -> pd.DataFrame
    def calculate_physics_features(self, df: pd.DataFrame) -> pd.DataFrame
    def create_final_dataset(self, df: pd.DataFrame) -> pd.DataFrame
    def upload_to_huggingface(self, df: pd.DataFrame, dataset_name: str)
```

#### Core Functions

##### Data Collection
```python
def download_nrel_data(self, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download solar irradiance and meteorological data from NREL API.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with columns:
        - timestamp: UTC datetime
        - Global_horizontal_irradiance: GHI in W/m²
        - Direct_normal_irradiance: DNI in W/m²
        - Diffuse_horizontal_irradiance: DHI in W/m²
        - Air_temperature: Temperature in °C
        - Wind_speed: Wind speed in m/s
        - Relative_humidity: Humidity in %
        - Pressure: Atmospheric pressure in hPa
    """
```

##### Image Processing
```python
def process_images(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Process satellite/sky images for each timestamp.
    
    Pipeline:
    1. Download images from satellite data sources
    2. Apply cloud detection and masking
    3. Crop and resize to standard format (224x224)
    4. Apply data augmentation for training robustness
    5. Quality filtering to remove corrupted images
    
    Args:
        df: DataFrame with timestamp information
        
    Returns:
        DataFrame with added columns:
        - image_path: Local path to processed image
        - image_quality_score: Quality metric (0-1)
        - cloud_coverage: Estimated cloud coverage (0-1)
    """
```

##### Physics Calculations
```python
def calculate_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive physics-informed features.
    
    Features calculated:
    - Solar position (zenith, azimuth angles)
    - Clear-sky irradiance estimates
    - Panel orientation optimization
    - Angle of incidence calculations
    - Atmospheric mass corrections
    
    Args:
        df: DataFrame with timestamp and location data
        
    Returns:
        DataFrame with physics features:
        - Zenith_angle: Solar zenith angle in degrees
        - Azimuth_angle: Solar azimuth angle in degrees
        - Clear_sky_ghi: Clear-sky GHI estimate in W/m²
        - physics_panel_tilt: Optimal panel tilt in degrees
        - physics_panel_orientation: Optimal panel azimuth in degrees
        - physics_aoi: Angle of incidence in degrees
        - physics_total_irradiance_tilted: Tilted plane irradiance in W/m²
    """
```

#### Usage Example
```python
from src.data_processing import SolarDataProcessor
from utils.helper import load_config

# Load configuration
config = load_config('config/config.yaml')

# Initialize processor
processor = SolarDataProcessor(config)

# Run complete pipeline
final_dataset = processor.process_complete_pipeline()

# Upload to HuggingFace
processor.upload_to_huggingface(final_dataset, config['huggingface']['dataset_name'])
```

---

### 2. `embeddings.py` - EmbeddingGenerator

**Purpose**: Extract high-dimensional feature representations from sky images using pre-trained Vision Transformers.

#### Key Classes

##### `EmbeddingGenerator`
Handles ViT-based feature extraction with efficient batch processing.

```python
class EmbeddingGenerator:
    """
    Vision Transformer-based embedding extraction for sky images.
    
    Uses pre-trained ViT-huge-patch14-224 model to extract 384-dimensional
    embeddings that capture semantic information about sky conditions.
    """
    
    def __init__(self, config: Dict[str, Any])
    def generate_embeddings(self, dataset_name: str, output_file: str)
    def process_image_batch(self, images: List[Image]) -> np.ndarray
    def save_embeddings(self, embeddings: List[Dict], output_file: str)
```

#### Core Functions

##### Embedding Generation
```python
def generate_embeddings(self, dataset_name: str, output_file: str):
    """
    Generate embeddings for all images in a dataset.
    
    Process:
    1. Load dataset from HuggingFace Hub
    2. Initialize pre-trained ViT model
    3. Process images in batches for memory efficiency
    4. Extract 384-dimensional feature vectors
    5. Save embeddings in JSON Lines format
    
    Args:
        dataset_name: HuggingFace dataset identifier
        output_file: Path to save embeddings (.json)
        
    Output Format:
        Each line: {"image_id": str, "embedding": List[float]}
    """
```

##### Batch Processing
```python
def process_image_batch(self, images: List[Image]) -> np.ndarray:
    """
    Process a batch of images through ViT model.
    
    Pipeline:
    1. Preprocessing: Resize, normalize, tensorize
    2. Forward pass through ViT encoder
    3. Extract CLS token representation
    4. Apply L2 normalization
    
    Args:
        images: List of PIL Images
        
    Returns:
        embeddings: (batch_size, 384) feature matrix
    """
```

#### Model Architecture Details
- **Base Model**: `google/vit-huge-patch14-224-in21k`
- **Input Size**: 224×224 RGB images
- **Output**: 384-dimensional embeddings
- **Preprocessing**: ImageNet normalization
- **Hardware**: Supports GPU acceleration

#### Usage Example
```python
from src.embeddings import EmbeddingGenerator

# Initialize generator
generator = EmbeddingGenerator(config)

# Generate embeddings for dataset
generator.generate_embeddings(
    dataset_name="username/solar-dataset",
    output_file="data/embeddings.json"
)

# Load embeddings for training
embeddings = []
with open("data/embeddings.json", "r") as f:
    for line in f:
        entry = json.loads(line)
        embeddings.append(entry["embedding"])
```

---

### 3. `nowcasting.py` - Modular Regression Framework

**Purpose**: Modular framework supporting multiple regression algorithms for real-time solar irradiance prediction.

#### Architecture Overview

The module implements a clean abstraction layer allowing easy switching between different regression models while maintaining consistent interfaces and training pipelines.

```
BaseRegressionModel (Abstract)
├── XGBoostModel
├── RandomForestModel  
└── MLPModel

ModelFactory → Creates appropriate model instances
NowcastingModel → Orchestrates training pipeline
```

#### Key Classes

##### `BaseRegressionModel` (Abstract Base Class)
```python
class BaseRegressionModel(ABC):
    """
    Abstract interface for all regression models.
    
    Ensures consistent behavior across different algorithms
    while allowing for model-specific optimizations.
    """
    
    @abstractmethod
    def get_optuna_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define hyperparameter search space for Optuna optimization."""
        
    @abstractmethod
    def create_model(self, params: Dict[str, Any]):
        """Create model instance with specified parameters."""
        
    @abstractmethod
    def fit(self, model, X_train, y_train, X_val=None, y_val=None):
        """Train model with optional validation set."""
        
    @abstractmethod
    def predict(self, model, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input features."""
        
    @abstractmethod
    def save_model(self, model, path: str):
        """Persist trained model to disk."""
        
    @abstractmethod
    def load_model(self, path: str):
        """Load trained model from disk."""
```

##### `XGBoostModel`
```python
class XGBoostModel(BaseRegressionModel):
    """
    XGBoost implementation optimized for solar irradiance prediction.
    
    Features:
    - Gradient boosting with tree ensembles
    - Early stopping to prevent overfitting
    - GPU acceleration support
    - Custom objective functions
    """
    
    def get_optuna_params(self, trial):
        """
        XGBoost-specific hyperparameter space:
        - max_depth: [3, 10] - Tree depth
        - learning_rate: [0.005, 0.3] - Step size
        - n_estimators: [100, 1500] - Number of trees
        - subsample: [0.5, 1.0] - Row sampling
        - colsample_bytree: [0.4, 1.0] - Column sampling
        - gamma: [0, 8] - Minimum split loss
        - lambda: [0, 8] - L2 regularization
        """
```

##### `RandomForestModel`
```python
class RandomForestModel(BaseRegressionModel):
    """
    Random Forest implementation with optimized hyperparameters.
    
    Features:
    - Ensemble of decision trees
    - Built-in feature importance
    - Robust to outliers
    - Parallel training support
    """
    
    def get_optuna_params(self, trial):
        """
        Random Forest hyperparameter space:
        - n_estimators: [50, 500] - Number of trees
        - max_depth: [3, 20] - Maximum tree depth
        - min_samples_split: [2, 20] - Minimum samples to split
        - min_samples_leaf: [1, 20] - Minimum samples per leaf
        - max_features: ["sqrt", "log2", None] - Feature sampling
        - bootstrap: [True, False] - Bootstrap sampling
        """
```

##### `MLPModel`
```python
class MLPModel(BaseRegressionModel):
    """
    Multi-Layer Perceptron with configurable architecture.
    
    Features:
    - 1-3 hidden layers with 50-500 neurons each
    - Multiple activation functions
    - Early stopping with validation
    - L2 regularization
    """
    
    def get_optuna_params(self, trial):
        """
        MLP hyperparameter space:
        - hidden_layer_sizes: Dynamic (1-3 layers, 50-500 neurons)
        - activation: ["relu", "tanh", "logistic"]
        - solver: ["adam", "lbfgs"]
        - alpha: [1e-5, 1e-1] (log scale) - L2 regularization
        - learning_rate: ["constant", "invscaling", "adaptive"]
        - learning_rate_init: [1e-4, 1e-1] (log scale)
        """
```

##### `ModelFactory`
```python
class ModelFactory:
    """
    Factory pattern for creating regression models.
    
    Supports:
    - 'xgboost': XGBoost gradient boosting
    - 'randomforest': Scikit-learn Random Forest
    - 'mlp': Scikit-learn Multi-Layer Perceptron
    """
    
    @classmethod
    def create_model(cls, model_type: str) -> BaseRegressionModel:
        """Create model instance by type identifier."""
        
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Return list of supported model types."""
```

##### `NowcastingModel`
```python
class NowcastingModel:
    """
    Main orchestration class for nowcasting model training.
    
    Handles:
    - Data loading and preprocessing
    - Feature engineering and normalization
    - Hyperparameter optimization
    - Model training and validation
    - Model persistence and loading
    """
    
    def __init__(self, config: Dict[str, Any])
    def load_data(self, dataset_name: str, embeddings_file: str)
    def train_final_model(self, dataset_name: str, embeddings_file: str, model_save_path: str)
    def load_trained_model(self, model_path: str, normalization_path: str = None)
    def predict_with_loaded_model(self, model, regression_model, embeddings, auxiliary_features)
```

#### Core Functions

##### Data Loading and Feature Engineering
```python
def load_data(self, dataset_name: str, embeddings_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and prepare data for training.
    
    Process:
    1. Load dataset from HuggingFace Hub
    2. Load pre-computed ViT embeddings
    3. Extract physics features with trigonometric transformations
    4. Combine embeddings with auxiliary features
    
    Feature Engineering:
    - Raw physics features (7): Clear-sky GHI, angles, tilt, AOI, tilted irradiance
    - Trigonometric features (10): sin/cos transformations of angular features
    - ViT embeddings (384): Pre-computed image features
    
    Args:
        dataset_name: HuggingFace dataset identifier
        embeddings_file: Path to ViT embeddings JSON file
        
    Returns:
        embeddings: (N, 384) ViT features
        auxiliary_features: (N, 17) physics features
        targets: (N, 1) GHI values in W/m²
    """
```

##### Training Pipeline
```python
def train_final_model(self, dataset_name: str, embeddings_file: str, model_save_path: str):
    """
    Complete training pipeline with hyperparameter optimization.
    
    Pipeline:
    1. Load and preprocess data
    2. Feature normalization (z-score)
    3. Hyperparameter optimization with Optuna
    4. Final model training on full dataset
    5. Model and normalization parameter saving
    
    Optimization:
    - Objective: Minimize validation nMAPE
    - Pruning: MedianPruner for early trial termination
    - Storage: SQLite for persistent study storage
    - Logging: Weights & Biases integration
    
    Args:
        dataset_name: HuggingFace dataset identifier
        embeddings_file: Path to embeddings file
        model_save_path: Output path for trained model
        
    Returns:
        final_model: Trained model instance
        best_params: Optimized hyperparameters
    """
```

##### Model Inference
```python
def predict_with_loaded_model(self, model, regression_model, embeddings, auxiliary_features) -> np.ndarray:
    """
    Generate predictions with loaded model.
    
    Pipeline:
    1. Combine embeddings with auxiliary features
    2. Apply stored normalization parameters
    3. Generate predictions using model
    4. Denormalize predictions to original scale
    
    Args:
        model: Loaded regression model
        regression_model: Model interface instance
        embeddings: (N, 384) ViT embeddings
        auxiliary_features: (N, 17) physics features
        
    Returns:
        predictions: (N,) GHI predictions in W/m²
    """
```

#### Usage Examples

##### Basic Training
```python
from src.nowcasting import NowcastingModel

# Load configuration
config = load_config('config/config.yaml')
config['nowcasting']['model_type'] = 'randomforest'

# Train model
model = NowcastingModel(config)
final_model, best_params = model.train_final_model(
    dataset_name="username/solar-dataset",
    embeddings_file="data/embeddings.json", 
    model_save_path="models/rf_model.pkl"
)
```

##### Model Comparison
```python
# Compare different models
models = ['xgboost', 'randomforest', 'mlp']
results = {}

for model_type in models:
    config['nowcasting']['model_type'] = model_type
    model = NowcastingModel(config)
    
    final_model, params = model.train_final_model(
        dataset_name, embeddings_file, f"models/{model_type}_model"
    )
    
    results[model_type] = params
```

---

### 4. `forecasting.py` - Temporal Forecasting Models

**Purpose**: Transformer-based architecture for predicting solar irradiance 1-4 hours into the future using temporal sequences.

#### Key Classes

##### `ForecastingModel`
```python
class ForecastingModel:
    """
    Transformer-based temporal forecasting for solar irradiance.
    
    Architecture:
    - Input: Sequence of embeddings from past 6 hours
    - Encoder: Multi-head attention with positional encoding
    - Decoder: Residual MLP for future predictions
    - Output: 15-minute interval predictions up to 4 hours
    """
    
    def __init__(self, config: Dict[str, Any])
    def prepare_temporal_data(self, embeddings: np.ndarray, targets: np.ndarray)
    def train_model(self, dataset_name: str, embeddings_file: str)
    def forecast(self, input_sequence: np.ndarray) -> np.ndarray
```

#### Core Functions

##### Temporal Data Preparation
```python
def prepare_temporal_data(self, embeddings: np.ndarray, targets: np.ndarray):
    """
    Convert static data into temporal sequences.
    
    Process:
    1. Create sliding windows of past observations
    2. Align with future target sequences
    3. Handle missing data and edge cases
    4. Apply temporal normalization
    
    Args:
        embeddings: (N, 384) ViT embeddings
        targets: (N,) GHI values
        
    Returns:
        sequences: (N_seq, seq_len, 384) input sequences
        targets: (N_seq, future_len) target sequences
    """
```

##### Model Architecture
```python
class TransformerForecastingModel(nn.Module):
    """
    Transformer encoder-decoder for solar forecasting.
    
    Components:
    - Positional Encoding: Sinusoidal encoding for temporal awareness
    - Multi-Head Attention: Self-attention for sequence modeling
    - Feed-Forward Networks: Non-linear transformations
    - Residual Connections: Skip connections for gradient flow
    - Layer Normalization: Stabilizes training
    """
    
    def __init__(self, input_dim=384, hidden_dim=512, num_heads=8, num_layers=6)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

#### Usage Example
```python
from src.forecasting import ForecastingModel

# Initialize forecasting model
forecaster = ForecastingModel(config)

# Train on temporal sequences
forecaster.train_model(
    dataset_name="username/solar-dataset",
    embeddings_file="data/embeddings.json"
)

# Generate forecasts
past_sequence = get_past_6_hours_embeddings()
future_predictions = forecaster.forecast(past_sequence)
```

## Integration and Data Flow

### Complete Pipeline Integration
```python
def run_complete_pipeline(config):
    """Example of running the complete SPIRIT pipeline."""
    
    # 1. Data Processing
    processor = SolarDataProcessor(config)
    dataset = processor.process_complete_pipeline()
    
    # 2. Embedding Generation
    generator = EmbeddingGenerator(config)
    generator.generate_embeddings(
        config['huggingface']['dataset_name'],
        config['embeddings']['output_file']
    )
    
    # 3. Nowcasting Model Training
    nowcaster = NowcastingModel(config)
    nowcast_model, params = nowcaster.train_final_model(
        config['huggingface']['dataset_name'],
        config['embeddings']['output_file'],
        config['training']['nowcast_model_path']
    )
    
    # 4. Forecasting Model Training
    forecaster = ForecastingModel(config)
    forecast_model = forecaster.train_model(
        config['huggingface']['dataset_name'],
        config['embeddings']['output_file']
    )
    
    return nowcast_model, forecast_model
```

### Data Format Standards

#### Dataset Schema
```python
# HuggingFace dataset columns:
{
    'timestamp': str,                    # ISO format UTC timestamp
    'Global_horizontal_irradiance': float, # Target GHI in W/m²
    'image_path': str,                   # Path to processed image
    'Zenith_angle': float,               # Solar zenith angle (degrees)
    'Azimuth_angle': float,              # Solar azimuth angle (degrees)
    'Clear_sky_ghi': float,              # Clear-sky GHI estimate (W/m²)
    'physics_panel_tilt': float,         # Optimal panel tilt (degrees)
    'physics_panel_orientation': float,   # Optimal panel azimuth (degrees)
    'physics_aoi': float,                # Angle of incidence (degrees)
    'physics_total_irradiance_tilted': float, # Tilted irradiance (W/m²)
    # Additional meteorological features...
}
```

#### Embedding Format
```json
# JSON Lines format (.json file):
{"image_id": "timestamp_001", "embedding": [0.123, -0.456, ...]}
{"image_id": "timestamp_002", "embedding": [0.789, -0.012, ...]}
```

For usage examples and detailed API documentation, see the main [README.md](../README.md) and individual script documentation in [scripts/README.md](../scripts/README.md).
generator = EmbeddingGenerator(config)
generator.process_dataset(dataset_name)
```

**Supported Models:** ViT-Base/Large/Huge, Resnet, etc

**Output:** JSON file with 768-1280 dimensional embeddings per image depending on the model chosen.

### `nowcasting.py` - NowcastingModel
XGBoost model for current solar irradiance prediction.

```python
model = NowcastingModel(config)
final_model, params = model.train_final_model(dataset, embeddings, save_path)
```

**Features:** Image embeddings + solar geometry + clear-sky calculations

**Output:** Current GHI prediction (W/m²)

### `forecasting.py` - ForecastingModel
Transformer model for 1-4 hour ahead prediction.

```python
model = ForecastingModel(config)
final_model, params = model.train_final_model(dataset, embeddings, save_path)
```

**Architecture:** Transformer encoder + residual MLPs

**Input:** 1-hour sequence of embeddings + future clear-sky values

**Output:** 24-step ahead GHI predictions - 4 hours.

## Quick Usage

```python
from src import SolarDataProcessor, EmbeddingGenerator, NowcastingModel, ForecastingModel

config = load_config('config/config.yaml')

# Complete pipeline
processor = SolarDataProcessor(config)
generator = EmbeddingGenerator(config)
nowcast = NowcastingModel(config)
forecast = ForecastingModel(config)
```

**Requirements:** All modules use `config/config.yaml` for configuration

**GPU Recommended:** For embeddings and forecasting modules
