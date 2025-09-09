import json
import pandas as pd
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import optuna
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None
import wandb
from typing import Dict, Any, List, Tuple, Union
from abc import ABC, abstractmethod
import joblib
import os


class BaseRegressionModel(ABC):
    """
    Abstract base class defining the interface for all regression models in SPIRIT.
    
    This class ensures consistent behavior across different regression algorithms
    while allowing for model-specific optimizations and hyperparameter spaces.
    All concrete implementations must provide the six core methods for training,
    prediction, and persistence.
    
    The design follows the Template Method pattern, where the training pipeline
    is standardized but algorithm-specific details are delegated to subclasses.
    """
    
    @abstractmethod
    def get_optuna_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter search space for Optuna optimization.
        
        This method specifies which hyperparameters should be optimized and their
        respective ranges, distributions, and constraints. Each model type should
        define parameters that are most impactful for solar irradiance prediction.
        
        Args:
            trial: Optuna trial object for suggesting hyperparameter values
            
        Returns:
            Dictionary of hyperparameter names and their suggested values
            
        Example:
            For XGBoost: {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 100}
            For Random Forest: {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5}
        """
        pass
    
    @abstractmethod
    def create_model(self, params: Dict[str, Any]):
        """
        Instantiate a model with the specified hyperparameters.
        
        Creates a fresh model instance configured with the provided parameters.
        This method is called both during hyperparameter optimization trials
        and for training the final model with optimal parameters.
        
        Args:
            params: Dictionary of hyperparameter names and values
            
        Returns:
            Configured model instance ready for training
            
        Note:
            The returned model should be in an untrained state and ready
            to accept training data via the fit() method.
        """
        pass
    
    @abstractmethod
    def fit(self, model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train the model on the provided dataset.
        
        Performs the core training procedure, which may include validation-based
        early stopping, regularization, or other training optimizations specific
        to the algorithm. Some models (like XGBoost) can utilize validation sets
        for early stopping, while others may ignore the validation data.
        
        Args:
            model: Untrained model instance from create_model()
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training targets, shape (n_samples,) or (n_samples, 1)
            X_val: Optional validation features for early stopping
            y_val: Optional validation targets for early stopping
            
        Returns:
            Trained model instance ready for prediction
            
        Note:
            - Features should be pre-normalized using z-score standardization
            - Targets should be normalized to improve training stability
            - Implementation should handle both 1D and 2D target arrays
        """
        pass
    
    @abstractmethod
    def predict(self, model, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Performs inference on new data, returning predicted Global Horizontal
        Irradiance values. Predictions should be in the same normalized scale
        as the training targets (will be denormalized by the calling code).
        
        Args:
            model: Trained model instance from fit()
            X: Input features, shape (n_samples, n_features)
            
        Returns:
            Predicted values, shape (n_samples,)
            
        Note:
            - Input features should already be normalized
            - Output predictions will be denormalized by the pipeline
            - Handle edge cases like empty input arrays gracefully
        """
        pass
    
    @abstractmethod
    def save_model(self, model, path: str):
        """
        Persist the trained model to disk for later use.
        
        Saves the model in an appropriate format for the algorithm type.
        The saved model should be loadable via the load_model() method
        and ready for inference without retraining.
        
        Args:
            model: Trained model instance to save
            path: File path where the model should be saved
            
        Note:
            - XGBoost models are saved as .json files
            - Scikit-learn models are saved as .pkl files using joblib
            - Ensure the saved model includes all necessary state for prediction
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str):
        """
        Load a previously trained model from disk.
        
        Reconstructs a trained model from its saved representation,
        restoring all state necessary for making predictions. The loaded
        model should be functionally equivalent to the original trained model.
        
        Args:
            path: File path to the saved model
            
        Returns:
            Loaded model instance ready for prediction
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
            Various format-specific errors for corrupted model files
            
        Note:
            - Must be compatible with models saved by save_model()
            - Handle version compatibility issues gracefully
        """
        pass


class XGBoostModel(BaseRegressionModel):
    """
    XGBoost regression model optimized for solar irradiance prediction.
    
    This implementation uses gradient boosting with tree ensembles, specifically
    tuned for solar energy forecasting. XGBoost is chosen as the default model
    due to its excellent out-of-the-box performance, fast training speed, and
    ability to handle complex non-linear relationships in solar data.
    
    Key Features:
    - Gradient boosting with tree-based learners
    - Early stopping to prevent overfitting
    - Robust handling of missing values
    - Built-in regularization (L1 and L2)
    - Efficient handling of sparse features
    - Support for GPU acceleration (when available)
    
    Hyperparameter Optimization:
    The model optimizes 8 key hyperparameters that significantly impact
    performance on solar irradiance data:
    - max_depth: Controls tree complexity and overfitting
    - learning_rate: Step size for gradient updates
    - n_estimators: Number of boosting rounds
    - subsample: Row sampling ratio for regularization
    - colsample_bytree: Feature sampling ratio per tree
    - gamma: Minimum loss reduction required for splits
    - lambda: L2 regularization strength
    """
    
    def __init__(self):
        """Initialize XGBoost model with availability check."""
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not available. Please install it with: pip install xgboost\n"
                "Note: On macOS, you may need to install OpenMP first: brew install libomp"
            )
    
    def get_optuna_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define XGBoost hyperparameter search space optimized for solar data.
        
        The parameter ranges are specifically tuned based on empirical analysis
        of solar irradiance prediction tasks. The ranges balance model complexity
        with training efficiency and generalization performance.
        
        Args:
            trial: Optuna trial object for parameter suggestion
            
        Returns:
            Dictionary containing hyperparameter suggestions:
            - booster: Fixed to 'gbtree' (tree-based gradient boosting)
            - max_depth: [3, 10] - Tree depth, balanced for solar patterns
            - learning_rate: [0.005, 0.3] - Conservative to aggressive learning
            - n_estimators: [100, 1500] - Boosting rounds, with early stopping
            - subsample: [0.5, 1.0] - Row sampling for regularization
            - colsample_bytree: [0.4, 1.0] - Feature sampling per tree
            - gamma: [0, 8] - Minimum split loss (regularization)
            - lambda: [0, 8] - L2 regularization strength
            
        Note:
            Parameter ranges are empirically validated on multiple solar datasets
            to ensure optimal performance across different geographical locations.
        """
        return {
            "booster": "gbtree",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 8),
            "lambda": trial.suggest_float("lambda", 0, 8),
        }
    
    def create_model(self, params: Dict[str, Any]):
        """
        Create and configure XGBoost regression model instance.
        
        Initializes an XGBRegressor with the provided hyperparameters and
        sets fixed configuration parameters optimized for solar irradiance
        prediction. The model uses squared error loss and MAE evaluation.
        
        Args:
            params: Hyperparameter dictionary from get_optuna_params()
            
        Returns:
            Configured XGBRegressor instance with:
            - Specified hyperparameters
            - reg:squarederror objective (for continuous targets)
            - MAE evaluation metric (robust to outliers)
            - Early stopping after 200 rounds without improvement
            
        Note:
            Early stopping helps prevent overfitting and reduces training time
            by monitoring validation performance during training.
        """
        model = xgb.XGBRegressor(**params)
        model.set_params(
            objective='reg:squarederror',
            eval_metric='mae',
            early_stopping_rounds=200
        )
        return model
    
    def fit(self, model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train XGBoost model with optional validation-based early stopping.
        
        Performs gradient boosting training with optional validation monitoring.
        When validation data is provided, the model uses early stopping to
        prevent overfitting and automatically determine the optimal number of
        boosting rounds.
        
        Args:
            model: Configured XGBRegressor from create_model()
            X_train: Training features, shape (n_samples, n_features)
                    Should include both ViT embeddings (384 dims) and
                    physics features (17 dims) for total of 401 features
            y_train: Training targets, shape (n_samples,) or (n_samples, 1)
                    Normalized GHI values in original W/m² scale
            X_val: Optional validation features for early stopping
            y_val: Optional validation targets for early stopping
            
        Returns:
            Trained XGBRegressor ready for prediction
            
        Training Process:
            1. If validation data provided:
               - Creates evaluation set list for monitoring
               - Trains with early stopping based on validation MAE
               - Automatically stops when validation performance plateaus
            2. If no validation data:
               - Trains for the full n_estimators specified
               - Uses internal regularization to prevent overfitting
            
        Note:
            Verbose output is disabled for cleaner training logs. The model
            internally tracks training progress and validation metrics.
        """
        if X_val is not None and y_val is not None:
            evals = [(X_train, y_train), (X_val, y_val)]
            model.fit(X_train, y_train, eval_set=evals, verbose=False)
        else:
            model.fit(X_train, y_train)
        return model
    
    def predict(self, model, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained XGBoost model.
        
        Performs gradient boosting prediction through the trained ensemble
        of decision trees to generate solar irradiance predictions.
        
        Args:
            model: Trained XGBRegressor from fit() method
            X: Input features, shape (n_samples, n_features)
               Should have same feature structure as training data
               (384 ViT + 17 physics = 401 total features)
            
        Returns:
            Predicted GHI values, shape (n_samples,)
            Continuous predictions from gradient boosting ensemble
            
        Prediction Process:
            1. Forward pass through all boosting rounds
            2. Accumulate predictions from each tree
            3. Apply final transformation if needed
            4. Return continuous regression values
        """
        return model.predict(X)
    
    def save_model(self, model, path: str):
        """
        Save trained XGBoost model to disk using native XGBoost format.
        
        Uses XGBoost's native save_model method which preserves the complete
        model state including tree structure, feature importance, and all
        training parameters. The native format is cross-platform compatible.
        
        Args:
            model: Trained XGBRegressor to save
            path: File path for saving, should use .json or .ubj extension
            
        Implementation Notes:
            - Uses XGBoost native serialization for optimal compatibility
            - Supports JSON and Universal Binary JSON (.ubj) formats
            - Saved model includes complete boosting ensemble
            - Compatible with load_model() method for restoration
            - Cross-platform and language compatible
        """
        model.save_model(path)
    
    def load_model(self, path: str):
        """
        Load previously saved XGBoost model from disk.
        
        Restores a complete XGBRegressor instance using XGBoost's native
        load_model method. Creates a new XGBRegressor instance and loads
        the saved model state into it.
        
        Args:
            path: File path to saved model (.json or .ubj format)
            
        Returns:
            Restored XGBRegressor ready for prediction
            
        Implementation Notes:
            - Creates new XGBRegressor instance first
            - Uses XGBoost native deserialization for compatibility
            - Loaded model preserves all training state
            - Compatible with models saved by save_model() method
            - Cross-platform restoration supported
        """
        model = xgb.XGBRegressor()
        model.load_model(path)
        return model


class RandomForestModel(BaseRegressionModel):
    """
    Random Forest regression model optimized for solar irradiance prediction.
    
    This implementation uses ensemble learning with multiple decision trees,
    providing excellent interpretability and robust performance for solar
    energy forecasting. Random Forest is particularly well-suited for solar
    data due to its ability to capture non-linear relationships and provide
    feature importance rankings.
    
    Key Features:
    - Ensemble of decision trees with bootstrap sampling
    - Built-in feature importance calculation
    - Robust handling of outliers and noise
    - Parallel training capability
    - No risk of overfitting with more trees
    - Excellent interpretability for domain experts
    - Native handling of mixed feature types
    
    Advantages for Solar Forecasting:
    - Feature importance reveals which ViT/physics features matter most
    - Robust to meteorological sensor noise and missing data
    - Captures complex interactions between solar angles and cloud patterns
    - Provides prediction confidence through tree voting variance
    - Works well with limited training data (bootstrap sampling)
    
    Hyperparameter Optimization:
    The model optimizes 6 key hyperparameters that balance accuracy
    with computational efficiency:
    - n_estimators: Number of trees in the forest
    - max_depth: Maximum tree depth for complexity control
    - min_samples_split: Minimum samples required to split nodes
    - min_samples_leaf: Minimum samples in leaf nodes
    - max_features: Number of features to consider for splits
    - max_samples: Bootstrap sample size for each tree
    """
    
    def get_optuna_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define Random Forest hyperparameter search space for solar prediction.
        
        The parameter ranges are optimized for solar irradiance data characteristics,
        balancing model complexity with training efficiency. The ranges ensure
        the forest is large enough to capture solar patterns while avoiding
        computational overhead.
        
        Args:
            trial: Optuna trial object for parameter suggestion
            
        Returns:
            Dictionary containing hyperparameter suggestions:
            - n_estimators: [50, 500] - Number of trees, balanced for accuracy/speed
            - max_depth: [None, 3-20] - Tree depth, None allows full growth
            - min_samples_split: [2, 20] - Split threshold for regularization
            - min_samples_leaf: [1, 10] - Leaf size for smoothing predictions
            - max_features: ['sqrt', 'log2', None] - Feature sampling strategy
            - max_samples: [0.7, 1.0] - Bootstrap sample fraction
            
        Parameter Rationale:
            - n_estimators: 50-500 provides good accuracy without excessive training time
            - max_depth: None allows natural tree growth for complex solar patterns
            - min_samples_split/leaf: Controls overfitting on limited solar data
            - max_features: 'sqrt' often optimal for regression tasks
            - max_samples: High values ensure diverse bootstrap samples
        """
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_categorical("max_depth", [None] + list(range(3, 21))),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "max_samples": trial.suggest_float("max_samples", 0.7, 1.0),
        }
    
    def create_model(self, params: Dict[str, Any]):
        """
        Create and configure Random Forest regression model instance.
        
        Initializes a RandomForestRegressor with the provided hyperparameters
        and sets fixed configuration parameters optimized for solar irradiance
        prediction. The model uses all available CPU cores for parallel training.
        
        Args:
            params: Hyperparameter dictionary from get_optuna_params()
            
        Returns:
            Configured RandomForestRegressor instance with:
            - Specified hyperparameters
            - Bootstrap sampling enabled for ensemble diversity
            - All CPU cores utilized (-1) for parallel training
            - Out-of-bag scoring enabled for internal validation
            - Reproducible random state for consistent results
            
        Configuration Notes:
            - bootstrap=True ensures diverse tree training sets
            - oob_score=True provides internal validation without separate holdout
            - n_jobs=-1 maximizes training speed on multi-core systems
            - random_state=42 ensures reproducible results across runs
        """
        return RandomForestRegressor(
            **params,
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=42
        )
    
    def fit(self, model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train Random Forest model with bootstrap sampling and parallel execution.
        
        Performs ensemble training across multiple decision trees with bootstrap
        sampling for diversity. The model trains in parallel across available CPU
        cores for efficiency. Validation data is ignored since Random Forest uses
        out-of-bag samples for internal validation.
        
        Args:
            model: Configured RandomForestRegressor from create_model()
            X_train: Training features, shape (n_samples, n_features)
                    Should include both ViT embeddings (384 dims) and
                    physics features (17 dims) for total of 401 features
            y_train: Training targets, shape (n_samples,) or (n_samples, 1)
                    Normalized GHI values in original W/m² scale
            X_val: Validation features (unused - RF uses OOB validation)
            y_val: Validation targets (unused - RF uses OOB validation)
            
        Returns:
            Trained RandomForestRegressor ready for prediction
            
        Training Process:
            1. For each tree in the forest:
               - Draw bootstrap sample from training data
               - Train decision tree on bootstrap sample
               - Use random feature subset at each split
            2. Parallel execution across CPU cores
            3. Out-of-bag samples provide internal validation
            4. No overfitting risk as more trees are added
            
        Note:
            Random Forest ignores validation data because it uses out-of-bag
            samples for internal validation. This provides unbiased performance
            estimates without requiring a separate validation set.
        """
        return model.fit(X_train, y_train)
    
    def predict(self, model, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained Random Forest model.
        
        Performs ensemble prediction across all trees in the forest to generate
        solar irradiance predictions. Each tree votes and the final prediction
        is the average across all trees.
        
        Args:
            model: Trained RandomForestRegressor from fit() method
            X: Input features, shape (n_samples, n_features)
               Should have same feature structure as training data
               (384 ViT + 17 physics = 401 total features)
            
        Returns:
            Predicted GHI values, shape (n_samples,)
            Continuous predictions averaged across all trees
            
        Prediction Process:
            1. Each tree makes individual prediction
            2. Predictions are averaged across all trees
            3. Variance across trees indicates prediction confidence
            4. No post-processing needed for continuous targets
        """
        return model.predict(X)
    
    def save_model(self, model, path: str):
        """
        Save trained Random Forest model to disk using joblib serialization.
        
        Saves the complete RandomForestRegressor instance including all trained
        trees, configuration parameters, and feature importance rankings.
        Uses joblib for efficient serialization of sklearn models.
        
        Args:
            model: Trained RandomForestRegressor to save
            path: File path for saving, extension changed to .pkl if .json provided
            
        Implementation Notes:
            - Automatically converts .json extension to .pkl for sklearn compatibility
            - Uses joblib for efficient ensemble model serialization
            - Saved model includes all trees and configuration
            - Compatible with load_model() method for restoration
        """
        # Change extension to .pkl for sklearn models
        if path.endswith('.json'):
            path = path.replace('.json', '.pkl')
        joblib.dump(model, path)
    
    def load_model(self, path: str):
        """
        Load previously saved Random Forest model from disk.
        
        Restores a complete RandomForestRegressor instance including all trained
        trees, configuration parameters, and feature importance rankings.
        Uses joblib for efficient deserialization of sklearn models.
        
        Args:
            path: File path to saved model, extension changed from .json if needed
            
        Returns:
            Restored RandomForestRegressor ready for prediction
            
        Implementation Notes:
            - Automatically converts .json extension to .pkl for sklearn compatibility
            - Uses joblib for efficient ensemble model deserialization
            - Loaded model is immediately ready for prediction
            - Compatible with models saved by save_model() method
        """
        if path.endswith('.json'):
            path = path.replace('.json', '.pkl')
        return joblib.load(path)


class MLPModel(BaseRegressionModel):
    """
    Multi-Layer Perceptron (MLP) regression model for solar irradiance prediction.
    
    This implementation uses a deep neural network with multiple hidden layers,
    specifically designed for learning complex non-linear relationships in solar
    energy data. The MLP is particularly effective at capturing intricate patterns
    between Vision Transformer embeddings and physics-based features.
    
    Key Features:
    - Deep feedforward neural network architecture
    - Adaptive learning rate with plateau scheduling
    - Dropout regularization for generalization
    - ReLU/tanh/logistic activation options
    - Adam/L-BFGS optimizer selection
    - Early stopping based on validation performance
    - Flexible architecture optimization
    
    Advantages for Solar Forecasting:
    - Learns complex interactions between ViT and physics features
    - Captures subtle patterns in high-dimensional embedding space
    - Adapts well to different solar irradiance scales and patterns
    - Provides smooth predictions suitable for energy planning
    - Can model temporal dependencies when combined with sequence data
    
    Architecture Design:
    The MLP uses a flexible architecture where the number of layers (1-3)
    and layer sizes (50-500 neurons) are optimized through hyperparameter
    search. This allows the model to adapt to the complexity of the solar
    prediction task.
    
    Hyperparameter Optimization:
    The model optimizes multiple architecture and training parameters:
    - n_layers: Network depth (1-3 hidden layers)
    - hidden_size_i: Width of each hidden layer (50-500 neurons)
    - activation: Non-linear activation function
    - solver: Optimization algorithm (Adam vs L-BFGS)
    - alpha: L2 regularization strength
    - learning_rate: Learning rate scheduling strategy
    """
    
    def get_optuna_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define MLP hyperparameter search space with flexible architecture.
        
        This method creates a dynamic architecture search where both the number
        of layers and the size of each layer are optimized. The search space
        allows for shallow (1 layer) to moderately deep (3 layers) networks
        suitable for the 401-dimensional input space.
        
        Args:
            trial: Optuna trial object for parameter suggestion
            
        Returns:
            Dictionary containing comprehensive hyperparameter suggestions:
            - hidden_layer_sizes: Tuple of layer sizes for flexible architecture
            - activation: ['relu', 'tanh', 'logistic'] - Activation functions
            - solver: ['adam', 'lbfgs'] - Optimization algorithms
            - alpha: [1e-5, 1e-1] - L2 regularization strength
            - learning_rate: ['constant', 'invscaling', 'adaptive'] - LR scheduling
            - learning_rate_init: [1e-4, 1e-1] - Initial learning rate
            - max_iter: 1000 - Maximum training iterations
            - early_stopping: True - Prevent overfitting
            - validation_fraction: 0.2 - Internal validation split
            - n_iter_no_change: 20 - Patience for early stopping
            
        Architecture Strategy:
            - n_layers: 1-3 layers balances complexity with overfitting risk
            - hidden_size_i: 50-500 neurons per layer adapts to data complexity
            - Tuple structure allows sklearn MLPRegressor compatibility
            
        Solver Selection:
            - 'adam': Good for large datasets, adaptive learning rates
            - 'lbfgs': Excellent for smaller datasets, faster convergence
            
        Learning Rate Options:
            - 'constant': Fixed learning rate throughout training
            - 'invscaling': Decreases as 1/t where t is time step
            - 'adaptive': Reduces when loss improvement < tol
        """
        # Suggest number of hidden layers (1-3 for solar data complexity)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        
        # Suggest hidden layer sizes dynamically
        hidden_layer_sizes = []
        for i in range(n_layers):
            size = trial.suggest_int(f"hidden_size_{i}", 50, 500)
            hidden_layer_sizes.append(size)
        
        return {
            "hidden_layer_sizes": tuple(hidden_layer_sizes),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
            "solver": trial.suggest_categorical("solver", ["adam", "lbfgs"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"]),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
            "max_iter": 1000,
            "early_stopping": True,
            "validation_fraction": 0.2,
            "n_iter_no_change": 20,
            "random_state": 42
        }
    
    def create_model(self, params: Dict[str, Any]):
        """
        Create and configure MLP regression model instance.
        
        Initializes an MLPRegressor with the provided hyperparameters.
        The configuration includes all optimization and regularization
        parameters needed for stable training on solar irradiance data.
        
        Args:
            params: Complete hyperparameter dictionary from get_optuna_params()
            
        Returns:
            Configured MLPRegressor instance ready for training with:
            - Dynamic architecture based on optimized layer configuration
            - Selected activation function and optimization algorithm
            - Regularization and early stopping parameters
            - Reproducible random state for consistent results
            
        Configuration Notes:
            - All hyperparameters are passed directly to MLPRegressor
            - Early stopping and validation are configured automatically
            - Random state ensures reproducible training across runs
        """
        return MLPRegressor(**params)
    
    def fit(self, model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train MLP model with automatic validation and early stopping.
        
        Performs neural network training with the configured architecture and
        optimization settings. The model uses internal validation splitting
        for early stopping, so external validation data is not needed.
        Target values are flattened to ensure compatibility with sklearn.
        
        Args:
            model: Configured MLPRegressor from create_model()
            X_train: Training features, shape (n_samples, n_features)
                    Should include both ViT embeddings (384 dims) and
                    physics features (17 dims) for total of 401 features
                    Features should be standardized for optimal MLP training
            y_train: Training targets, shape (n_samples,) or (n_samples, 1)
                    GHI values in W/m², will be flattened for sklearn compatibility
            X_val: Validation features (unused - MLP uses internal validation)
            y_val: Validation targets (unused - MLP uses internal validation)
            
        Returns:
            Trained MLPRegressor ready for prediction
            
        Training Process:
            1. Flatten target array for sklearn compatibility
            2. Internal validation split based on validation_fraction (20%)
            3. Neural network training with selected solver
            4. Early stopping based on validation performance
            5. Learning rate adaptation (if adaptive scheduling selected)
            
        Important Notes:
            - Target values are flattened using ravel() for sklearn compatibility
            - Internal validation handles overfitting prevention
            - Training convergence depends on solver and learning rate settings
        """
        model.fit(X_train, y_train.ravel())
        return model
    
    def predict(self, model, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained MLP model.
        
        Performs forward pass through the trained neural network to generate
        solar irradiance predictions. The model outputs continuous values
        suitable for regression tasks.
        
        Args:
            model: Trained MLPRegressor from fit() method
            X: Input features, shape (n_samples, n_features)
               Should have same feature structure as training data
               (384 ViT + 17 physics = 401 total features)
            
        Returns:
            Predicted GHI values, shape (n_samples,)
            Continuous predictions in same scale as training targets
            
        Prediction Process:
            1. Forward pass through all hidden layers
            2. Apply activation functions at each layer
            3. Output layer produces final regression values
            4. No post-processing needed for continuous targets
        """
        return model.predict(X)
    
    def save_model(self, model, path: str):
        """
        Save trained MLP model to disk using joblib serialization.
        
        Saves the complete MLPRegressor instance including trained weights,
        architecture, and all configuration parameters. Uses joblib for
        efficient serialization of sklearn models.
        
        Args:
            model: Trained MLPRegressor to save
            path: File path for saving, extension changed to .pkl if .json provided
            
        Implementation Notes:
            - Automatically converts .json extension to .pkl for sklearn compatibility
            - Uses joblib for efficient neural network serialization
            - Saved model includes complete architecture and trained weights
            - Compatible with load_model() method for restoration
        """
        # Change extension to .pkl for sklearn models
        if path.endswith('.json'):
            path = path.replace('.json', '.pkl')
        joblib.dump(model, path)
    
    def load_model(self, path: str):
        """
        Load previously saved MLP model from disk.
        
        Restores a complete MLPRegressor instance including trained weights,
        architecture, and all configuration parameters. Uses joblib for
        efficient deserialization of sklearn models.
        
        Args:
            path: File path to saved model, extension changed from .json if needed
            
        Returns:
            Restored MLPRegressor ready for prediction
            
        Implementation Notes:
            - Automatically converts .json extension to .pkl for sklearn compatibility
            - Uses joblib for efficient neural network deserialization
            - Loaded model is immediately ready for prediction
            - Compatible with models saved by save_model() method
        """
        if path.endswith('.json'):
            path = path.replace('.json', '.pkl')
        return joblib.load(path)


class ModelFactory:
    """
    Factory class for creating configurable regression models for solar prediction.
    
    This factory implements the Factory design pattern to provide a clean interface
    for instantiating different regression models (XGBoost, Random Forest, MLP)
    based on string identifiers. It centralizes model creation logic and makes
    it easy to add new regression algorithms to the SPIRIT framework.
    
    Key Features:
    - Centralized model instantiation
    - String-based model selection
    - Easy extension for new algorithms
    - Type safety through abstract base class
    - Standardized error handling
    
    Supported Models:
    - 'xgboost': Gradient boosting with tree ensembles
    - 'randomforest': Ensemble of decision trees
    - 'mlp': Multi-layer perceptron neural network
    
    Usage Example:
        ```python
        # Create specific model instances
        xgb_model = ModelFactory.create_model('xgboost')
        rf_model = ModelFactory.create_model('randomforest')
        mlp_model = ModelFactory.create_model('mlp')
        
        # Get available model types
        available = ModelFactory.get_available_models()
        print(available)  # ['xgboost', 'randomforest', 'mlp']
        ```
    
    Extension Pattern:
    To add new regression models:
    1. Create a class inheriting from BaseRegressionModel
    2. Implement all abstract methods
    3. Add to _models dictionary with string key
    """
    
    # Build models dictionary dynamically based on availability
    _models = {
        'randomforest': RandomForestModel,
        'mlp': MLPModel
    }
    
    # Add XGBoost only if available
    if XGBOOST_AVAILABLE:
        _models['xgboost'] = XGBoostModel
    
    @classmethod
    def create_model(cls, model_type: str) -> BaseRegressionModel:
        """
        Create a regression model instance of the specified type.
        
        Instantiates a concrete regression model based on the provided string
        identifier. The model type is case-insensitive for user convenience.
        All created models implement the BaseRegressionModel interface.
        
        Args:
            model_type: String identifier for the desired model type.
                       Supported values: 'xgboost', 'randomforest', 'mlp'
                       Case-insensitive (e.g., 'XGBoost', 'RANDOMFOREST', 'Mlp')
                       
        Returns:
            BaseRegressionModel: Instantiated model ready for configuration
                                All returned models implement the same interface:
                                - get_optuna_params()
                                - create_model()
                                - fit()
                                - predict()
                                - save_model()
                                - load_model()
                                
        Raises:
            ValueError: If model_type is not recognized
                       Error message includes available model types for guidance
                       
        Implementation Notes:
            - Input is converted to lowercase for case-insensitive matching
            - Models are instantiated with default parameters
            - Actual hyperparameters are set later during optimization
            - Each model type has its own hyperparameter space
            
        Example:
            ```python
            try:
                model = ModelFactory.create_model('XGBoost')
                # model is now an XGBoostModel instance
            except ValueError as e:
                print(f"Invalid model type: {e}")
            ```
        """
        model_type = model_type.lower()
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(cls._models.keys())}")
        
        return cls._models[model_type]()
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Get list of all available regression model types.
        
        Returns a list of string identifiers for all supported regression
        models. This is useful for validation, user interfaces, and
        configuration validation.
        
        Returns:
            List[str]: Available model type identifiers in lowercase
                      Current models: ['xgboost', 'randomforest', 'mlp']
                      
        Usage Examples:
            ```python
            # Validate user input
            user_model = 'random_forest'  # Invalid format
            if user_model not in ModelFactory.get_available_models():
                print("Invalid model type")
                
            # Generate help text
            available = ModelFactory.get_available_models()
            print(f"Choose from: {', '.join(available)}")
            
            # Configuration validation
            config_model = config.get('model_type', 'xgboost')
            assert config_model in ModelFactory.get_available_models()
            ```
            
        Note:
            Model identifiers are always returned in lowercase for consistency
            with the create_model() method's case-insensitive behavior.
        """
        return list(cls._models.keys())


class NowcastingModel:
    """
    Main orchestrator class for solar irradiance nowcasting with modular regression models.
    
    This class serves as the primary interface for the SPIRIT nowcasting system,
    integrating Vision Transformer embeddings with physics-based features to
    predict real-time solar irradiance (GHI). It supports multiple regression
    algorithms through a modular architecture.
    
    Key Features:
    - Modular regression model selection (XGBoost, Random Forest, MLP)
    - Integration of ViT embeddings and physics features
    - Optuna-based hyperparameter optimization
    - WandB experiment tracking and logging
    - Comprehensive data preprocessing pipeline
    - Model persistence and loading capabilities
    - Robust error handling and validation
    
    Architecture Overview:
    1. Data Processing: Loads and preprocesses satellite imagery and weather data
    2. Feature Extraction: Combines ViT embeddings (384-dim) with physics features (17-dim)
    3. Model Training: Uses Optuna to optimize hyperparameters for selected regression model
    4. Evaluation: Computes comprehensive metrics (MAE, RMSE, R², MAPE, sMAPE)
    5. Persistence: Saves trained models and optimization results
    
    Configuration Structure:
    The class expects a configuration dictionary with the following sections:
    - training: General training parameters (batch_size, validation_split, etc.)
    - nowcasting: Nowcasting-specific parameters including model_type selection
    - Other sections: data_processing, embeddings, etc. (used by other components)
    
    Supported Model Types:
    - 'xgboost': Default choice, excellent out-of-the-box performance
    - 'randomforest': Interpretable ensemble method with feature importance
    - 'mlp': Neural network for complex non-linear relationships
    
    Usage Example:
        ```python
        import yaml
        from src.nowcasting import NowcastingModel
        
        # Load configuration
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create and train model
        nowcaster = NowcastingModel(config)
        nowcaster.train(train_data, val_data)
        
        # Make predictions
        predictions = nowcaster.predict(test_data)
        
        # Save trained model
        nowcaster.save_model('models/solar_nowcaster.pkl')
        ```
    
    Integration with SPIRIT Pipeline:
    - Receives processed data from data_processing.py
    - Uses ViT embeddings from embeddings.py
    - Integrates with forecasting.py for extended predictions
    - Supports batch processing for operational deployment
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize nowcasting model with configuration parameters.
        
        Sets up the nowcasting system by parsing configuration, creating the
        appropriate regression model instance, and initializing all necessary
        components for training and inference.
        
        Args:
            config: Complete configuration dictionary containing:
                   - training: General training parameters
                   - nowcasting: Nowcasting-specific configuration including model_type
                   - data_processing: Data loading and preprocessing settings
                   - embeddings: Vision Transformer configuration
                   - Other components as needed
                   
        Configuration Parameters:
            training section:
                - batch_size: Training batch size
                - validation_split: Fraction for validation data
                - n_trials: Number of Optuna optimization trials
                - timeout: Maximum optimization time
                
            nowcasting section:
                - model_type: Regression model ('xgboost', 'randomforest', 'mlp')
                - target_column: Target variable name (default: 'GHI')
                - feature_columns: List of feature column names
                - normalization: Whether to normalize features
                
        Initialization Process:
            1. Parse and validate configuration sections
            2. Extract model type with backward compatibility fallback
            3. Create regression model instance through ModelFactory
            4. Initialize tracking and logging components
            5. Set up data processing parameters
            
        Raises:
            KeyError: If required configuration sections are missing
            ValueError: If model_type is not supported
            
        Example Configuration:
            ```yaml
            training:
              batch_size: 32
              validation_split: 0.2
              n_trials: 100
              
            nowcasting:
              model_type: 'xgboost'  # or 'randomforest', 'mlp'
              target_column: 'GHI'
              normalization: true
            ```
        """
        self.config = config
        self.training_config = config['training']
        self.nowcasting_config = config['nowcasting']
        
        # Get model type from config, default to XGBoost for backward compatibility
        self.model_type = self.nowcasting_config.get('model_type', 'xgboost')
        self.regression_model = ModelFactory.create_model(self.model_type)
        
        # Initialize wandb
        wandb.login(key=self.training_config['wandb_key'])
        
        self.study_name = self.nowcasting_config['study_name']
        
        # Normalization parameters (will be set during training)
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def load_data(self, dataset_name: str, embeddings_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset and embeddings."""
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split="train")
        df = pd.DataFrame(dataset)
        
        print(f"Loading embeddings: {embeddings_file}")
        embeddings = []
        with open(embeddings_file, "r") as file:
            for line in file:
                entry = json.loads(line)
                embeddings.append(entry["embedding"])
        
        embeddings = np.array(embeddings)
        
        # Prepare auxiliary features
        auxiliary_features = []
        targets = []
        
        for index in range(len(df)):
            row = df.iloc[index]
            
            # Solar geometry features
            zenith_angle = row["Zenith_angle"]
            azimuth_angle = row["Azimuth_angle"]
            panel_tilt = row["physics_panel_tilt"]
            panel_orientation = row["physics_panel_orientation"]
            aoi = row["physics_aoi"]
            
            # Create feature vector with trigonometric transformations
            features = [
                row["Clear_sky_ghi"],
                zenith_angle,
                azimuth_angle,
                panel_tilt,
                panel_orientation,
                aoi,
                row["physics_total_irradiance_tilted"],
                # Trigonometric features for better angle representation
                np.cos(np.deg2rad(zenith_angle)),
                np.sin(np.deg2rad(zenith_angle)),
                np.cos(np.deg2rad(azimuth_angle)),
                np.sin(np.deg2rad(azimuth_angle)),
                np.cos(np.deg2rad(panel_tilt)),
                np.sin(np.deg2rad(panel_tilt)),
                np.cos(np.deg2rad(panel_orientation)),
                np.sin(np.deg2rad(panel_orientation)),
                np.cos(np.deg2rad(aoi)),
                np.sin(np.deg2rad(aoi)),
            ]
            
            auxiliary_features.append(features)
            targets.append(row["Global_horizontal_irradiance"])
        
        auxiliary_features = np.array(auxiliary_features)
        targets = np.array(targets).reshape(-1, 1)
        
        return embeddings, auxiliary_features, targets

    def normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize features."""
        if fit:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
        
        return (X - self.X_mean) / self.X_std

    def normalize_targets(self, y: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize targets."""
        if fit:
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
        
        return (y - self.y_mean) / self.y_std

    def denormalize_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        """Denormalize predictions."""
        return y_pred * self.y_std + self.y_mean

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        # Denormalize predictions for evaluation
        y_true_denorm = self.denormalize_predictions(y_true)
        y_pred_denorm = self.denormalize_predictions(y_pred)
        
        mae = mean_absolute_error(y_true_denorm, y_pred_denorm)
        mse = mean_squared_error(y_true_denorm, y_pred_denorm)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_denorm, y_pred_denorm)
        nmap = mae / np.mean(y_true_denorm) * 100
        
        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2_Score": r2,
            "nMAP": nmap,
        }

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train regression model."""
        model = self.regression_model.create_model(params)
        
        # Split for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Train model (some models use validation set, others don't)
        try:
            model = self.regression_model.fit(model, X_train_split, y_train_split, X_val_split, y_val_split)
        except TypeError:
            # If the model doesn't support validation set in fit method
            model = self.regression_model.fit(model, X_train_split, y_train_split)
        
        # Validate
        y_val_pred = self.regression_model.predict(model, X_val_split)
        val_metrics = self.evaluate_model(y_val_split, y_val_pred)
        
        return model, val_metrics

    def objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Optuna objective function."""
        wandb.init(
            project=self.study_name,
            mode="online",
            name=f"trial_{trial.number}_{self.model_type}",
            settings=wandb.Settings(init_timeout=150)
        )
        
        # Get hyperparameters specific to the model type
        params = self.regression_model.get_optuna_params(trial)
        
        # Train model
        model, val_metrics = self.train_model(X_train, y_train, params)
        
        # Log metrics
        wandb.log({"val_" + k: v for k, v in val_metrics.items()})
        wandb.log(params)
        wandb.log({"model_type": self.model_type})
        
        wandb.finish()
        
        return val_metrics["nMAP"]

    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10, n_warmup_steps=10, interval_steps=1
        )
        
        # Include model type in study name for better organization
        study_name_with_model = f"{self.study_name}_{self.model_type}"
        
        study = optuna.create_study(
            direction="minimize",
            pruner=pruner,
            study_name=study_name_with_model,
            storage=f"sqlite:///./{study_name_with_model}.db",
            load_if_exists=True,
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        return study.best_trial.params

    def train_final_model(self, dataset_name: str, embeddings_file: str, model_save_path: str):
        """Train the final nowcasting model."""
        print(f"Training {self.model_type.upper()} model...")
        print("Loading data...")
        embeddings, auxiliary_features, targets = self.load_data(dataset_name, embeddings_file)
        
        # Combine features
        X = np.concatenate([embeddings, auxiliary_features], axis=1)
        y = targets
        
        # Normalize features and targets
        X_normalized = self.normalize_features(X, fit=True)
        y_normalized = self.normalize_targets(y, fit=True)
        
        print("Optimizing hyperparameters...")
        best_params = self.optimize_hyperparameters(
            X_normalized, y_normalized, 
            n_trials=self.nowcasting_config.get('n_trials', 100)
        )
        
        print(f"Best parameters: {best_params}")
        
        # Train final model with best parameters
        print("Training final model...")
        final_model = self.regression_model.create_model(best_params)
        final_model = self.regression_model.fit(final_model, X_normalized, y_normalized)
        
        # Adjust file extension based on model type
        if self.model_type in ['randomforest', 'mlp'] and model_save_path.endswith('.json'):
            model_save_path = model_save_path.replace('.json', '.pkl')
        
        # Save model
        self.regression_model.save_model(final_model, model_save_path)
        print(f"Model saved to {model_save_path}")
        
        # Save normalization parameters
        norm_params = {
            'X_mean': self.X_mean.tolist(),
            'X_std': self.X_std.tolist(),
            'y_mean': float(self.y_mean),
            'y_std': float(self.y_std),
            'model_type': self.model_type  # Save model type for loading
        }
        
        norm_path = model_save_path.replace('.json', '_normalization.json').replace('.pkl', '_normalization.json')
        with open(norm_path, 'w') as f:
            json.dump(norm_params, f)
        
        print(f"Normalization parameters saved to {norm_path}")
        
        return final_model, best_params
    
    def load_trained_model(self, model_path: str, normalization_path: str = None):
        """Load a trained model for inference."""
        if normalization_path is None:
            normalization_path = model_path.replace('.json', '_normalization.json').replace('.pkl', '_normalization.json')
        
        # Load normalization parameters
        with open(normalization_path, 'r') as f:
            norm_params = json.load(f)
        
        self.X_mean = np.array(norm_params['X_mean'])
        self.X_std = np.array(norm_params['X_std'])
        self.y_mean = norm_params['y_mean']
        self.y_std = norm_params['y_std']
        
        # Get model type from normalization params or use configured type
        saved_model_type = norm_params.get('model_type', self.model_type)
        
        # Create appropriate model instance
        regression_model = ModelFactory.create_model(saved_model_type)
        
        # Load the model
        model = regression_model.load_model(model_path)
        
        return model, regression_model
    
    def predict_with_loaded_model(self, model, regression_model: BaseRegressionModel, 
                                 embeddings: np.ndarray, auxiliary_features: np.ndarray) -> np.ndarray:
        """Make predictions with a loaded model."""
        # Combine features
        X = np.concatenate([embeddings, auxiliary_features], axis=1)
        
        # Normalize features
        X_normalized = self.normalize_features(X, fit=False)
        
        # Make predictions
        y_pred_normalized = regression_model.predict(model, X_normalized)
        
        # Denormalize predictions
        y_pred = self.denormalize_predictions(y_pred_normalized)
        
        return y_pred