import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import wandb
import pandas as pd
import json
from datasets import load_dataset
from typing import Dict, Any, List, Tuple
import os
from tqdm import tqdm


class ResidualMLP(nn.Module):
    """Residual MLP block."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x + residual


class TransformerForecastingModel(nn.Module):
    """Transformer-based forecasting model."""
    
    def __init__(self, feature_dim: int, hidden_dim: int, num_attention_heads: int, 
                 num_encoder_layers: int, num_residual_blocks: int, 
                 num_past_frames: int, num_future_frames: int):
        super().__init__()
        
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.initial_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_past_frames + 1, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        
        self.concatenation_fc_layer = nn.Linear(
            hidden_dim + num_future_frames, hidden_dim
        )
        
        self.residual_mlp = nn.Sequential(
            *[ResidualMLP(hidden_dim) for _ in range(num_residual_blocks)]
        )
        
        self.final_fc_layer = nn.Linear(hidden_dim, num_future_frames)

    def forward(self, x: torch.Tensor, future_clearsky_ghi: torch.Tensor) -> torch.Tensor:
        B, _, _ = x.shape
        
        # Project input features
        x = self.input_projection(x)
        
        # Add initial token
        initial_token = self.initial_token.expand(B, -1, -1).to(x.device)
        x = torch.cat([initial_token, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embedding.to(x.device)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Extract global representation (first token)
        x = x[:, 0, :]
        
        # Concatenate with future clear-sky information
        x = torch.cat([x, future_clearsky_ghi], dim=-1)
        x = self.concatenation_fc_layer(x)
        
        # Residual MLP processing
        x = self.residual_mlp(x)
        
        # Final prediction
        x = self.final_fc_layer(x)
        
        # Add clear-sky as residual connection
        x = x + future_clearsky_ghi
        
        return x


class ForecastingModel:
    """Solar irradiance forecasting using transformer architecture."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_config = config['training']
        self.forecasting_config = config['forecasting']
        
        # Initialize wandb
        wandb.login(key=self.training_config['wandb_key'])
        
        self.study_name = self.forecasting_config['study_name']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        self.num_past_frames = self.forecasting_config['num_past_frames']
        self.num_future_frames = self.forecasting_config['num_future_frames']
        self.batch_size = self.forecasting_config['batch_size']
        self.num_epochs = self.forecasting_config['num_epochs']
        
        print(f"Using device: {self.device}")

    def load_data(self, dataset_name: str, embeddings_file: str) -> Tuple[pd.DataFrame, List[List[float]]]:
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
        
        return df, embeddings

    def generate_timeseries_data(self, df: pd.DataFrame, embeddings: List[List[float]]) -> List[Tuple]:
        """Generate time series sequences for training."""
        timeseries_data = []
        len_dataset = len(embeddings)
        
        for i in range(len_dataset - self.num_past_frames - self.num_future_frames):
            # Check if all frames are from the same date
            if (df.iloc[i]["DATE"] != 
                df.iloc[i + self.num_past_frames + self.num_future_frames - 1]["DATE"]):
                continue
            
            # Past embeddings
            past_embeddings = [
                np.array(embeddings[j]) for j in range(i, i + self.num_past_frames)
            ]
            
            # Future clear-sky GHI values
            future_clear_sky_ghi = [
                df.iloc[i + self.num_past_frames + j]["Clear_sky_ghi"]
                for j in range(self.num_future_frames)
            ]
            
            # Target actual GHI values
            target_actual_ghi = [
                df.iloc[i + self.num_past_frames + j]["Global_horizontal_irradiance"]
                for j in range(self.num_future_frames)
            ]
            
            timeseries_data.append((past_embeddings, future_clear_sky_ghi, target_actual_ghi))
        
        return timeseries_data

    def prepare_tensors(self, timeseries_data: List[Tuple], split_data: bool = True) -> Dict[str, torch.Tensor]:
        """Convert timeseries data to tensors."""
        if split_data:
            # Split into train and validation
            X, X_future_clear_sky, y_ghi = zip(*timeseries_data)
            X_train, X_val, X_future_clear_sky_train, X_future_clear_sky_val, y_ghi_train, y_ghi_val = train_test_split(
                list(X), list(X_future_clear_sky), list(y_ghi), test_size=0.2, random_state=42
            )
            
            # Convert to tensors
            return {
                "X_train": torch.tensor(np.array(X_train), dtype=torch.float32),
                "X_val": torch.tensor(np.array(X_val), dtype=torch.float32),
                "X_future_clear_sky_train": torch.tensor(np.array(X_future_clear_sky_train), dtype=torch.float32),
                "X_future_clear_sky_val": torch.tensor(np.array(X_future_clear_sky_val), dtype=torch.float32),
                "y_ghi_train": torch.tensor(np.array(y_ghi_train), dtype=torch.float32),
                "y_ghi_val": torch.tensor(np.array(y_ghi_val), dtype=torch.float32),
            }
        else:
            X, X_future_clear_sky, y_ghi = zip(*timeseries_data)
            return {
                "X": torch.tensor(np.array(X), dtype=torch.float32),
                "X_future_clear_sky": torch.tensor(np.array(X_future_clear_sky), dtype=torch.float32),
                "y_ghi": torch.tensor(np.array(y_ghi), dtype=torch.float32),
            }

    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        warmup_steps = 500
        total_steps = 50000
        
        def warmup_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=0
        )
        
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    def save_model_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                            scheduler: torch.optim.lr_scheduler._LRScheduler, 
                            epoch: int, best_val_loss: float, save_path: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        }
        torch.save(checkpoint, save_path)

    def load_model_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                            scheduler: torch.optim.lr_scheduler._LRScheduler,
                            load_path: str) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, int, float]:
        """Load model checkpoint."""
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return model, optimizer, scheduler, checkpoint["epoch"], checkpoint["best_val_loss"]

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        nmap = mae / np.mean(y_true)
        std_dev = np.std(y_true - y_pred)
        
        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "nMAP": nmap,
            "std": std_dev,
        }

    def train_model(self, tensors: Dict[str, torch.Tensor], params: Dict[str, Any], 
                   trial: optuna.Trial = None) -> nn.Module:
        """Train the forecasting model."""
        # Get feature dimension from embeddings
        feature_dim = tensors["X_train"].shape[-1]
        
        # Create model
        model = TransformerForecastingModel(
            feature_dim=feature_dim,
            hidden_dim=params["hidden_dim"],
            num_attention_heads=params["num_attention_heads"],
            num_encoder_layers=params["num_encoder_layers"],
            num_residual_blocks=params["num_residual_blocks"],
            num_past_frames=self.num_past_frames,
            num_future_frames=self.num_future_frames
        ).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(
            tensors["X_train"], tensors["X_future_clear_sky_train"], tensors["y_ghi_train"]
        )
        val_dataset = TensorDataset(
            tensors["X_val"], tensors["X_future_clear_sky_val"], tensors["y_ghi_val"]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Setup training
        criterion = nn.L1Loss()
        optimizer = SGD(model.parameters(), lr=params["init_learning_rate"], momentum=0.9)
        scheduler = self.get_scheduler(optimizer)
        
        best_val_loss = float("inf")
        patience = 20
        epochs_no_improve = 0
        
        model_save_path = f"temp_model_{trial.number if trial else 'final'}.pth"
        
        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for X_batch, X_future_clear_sky_batch, y_ghi_batch in train_loader:
                X_batch = X_batch.to(self.device)
                X_future_clear_sky_batch = X_future_clear_sky_batch.to(self.device)
                y_ghi_batch = y_ghi_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Focus on specific horizons for training (1hr, 2hr, 3hr, 4hr)
                indices = [5, 11, 17, 23]
                y_pred = model(X_batch, X_future_clear_sky_batch)
                y_pred_selected = y_pred[:, indices]
                y_ghi_selected = y_ghi_batch[:, indices]
                
                loss = criterion(y_pred_selected, y_ghi_selected)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, X_future_clear_sky_batch, y_ghi_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    X_future_clear_sky_batch = X_future_clear_sky_batch.to(self.device)
                    y_ghi_batch = y_ghi_batch.to(self.device)
                    y_pred = model(X_batch, X_future_clear_sky_batch)
                    
                    indices = [5, 11, 17, 23]
                    y_pred_selected = y_pred[:, indices]
                    y_ghi_selected = y_ghi_batch[:, indices]
                    
                    loss = criterion(y_pred_selected, y_ghi_selected)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Log metrics
            if trial:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss})
            
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Pruning for optuna
            if trial:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    print(f"Pruned at epoch {epoch + 1}")
                    raise optuna.TrialPruned()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, model_save_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        model, _, _, _, _ = self.load_model_checkpoint(model, optimizer, scheduler, model_save_path)
        
        return model
    
    def objective(self, trial: optuna.Trial, tensors: Dict[str, torch.Tensor]) -> float:
        """Optuna objective function."""
        torch.cuda.empty_cache()
        
        wandb.init(
            project=self.study_name,
            mode="online",
            name=f"trial_{trial.number}",
        )
        
        # Suggest hyperparameters
        params = {
            "hidden_dim": trial.suggest_int("hidden_dim", 528, 2064, step=48),
            "num_residual_blocks": trial.suggest_int("num_residual_blocks", 3, 10),
            "num_encoder_layers": trial.suggest_int("num_encoder_layers", 8, 16),
            "init_learning_rate": trial.suggest_float("init_learning_rate", 1e-4, 1e-3, log=True),
            "num_attention_heads": trial.suggest_int("num_attention_heads", 8, 16, step=4),
        }
        
        wandb.config.update(params, allow_val_change=True)
        
        # Train model
        model = self.train_model(tensors, params, trial)
        
        # Evaluate on validation set
        val_metrics = self.evaluate_trained_model(model, tensors, validation=True)
        
        # Log metrics
        wandb.log(val_metrics)
        wandb.finish()
        
        # Return average nMAP for optimization
        nmap_values = [v for k, v in val_metrics.items() if "nMAP" in k]
        return sum(nmap_values) / len(nmap_values) if nmap_values else val_metrics.get("nMAP", float('inf'))

    def evaluate_trained_model(self, model: nn.Module, tensors: Dict[str, torch.Tensor], validation: bool = False) -> Dict[str, float]:
        """Evaluate trained model."""
        if validation:
            X_tensor = tensors["X_val"]
            X_future_clear_sky_tensor = tensors["X_future_clear_sky_val"]
            y_ghi_tensor = tensors["y_ghi_val"]
        else:
            X_tensor = tensors["X"]
            X_future_clear_sky_tensor = tensors["X_future_clear_sky"]
            y_ghi_tensor = tensors["y_ghi"]
        
        # Create dataloader
        dataset = TensorDataset(X_tensor, X_future_clear_sky_tensor, y_ghi_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        model.eval()
        
        y_true_by_step = [[] for _ in range(self.num_future_frames)]
        y_pred_by_step = [[] for _ in range(self.num_future_frames)]
        
        with torch.no_grad():
            for X_batch, X_future_clear_sky_batch, y_ghi_batch in dataloader:
                X_batch = X_batch.to(self.device)
                X_future_clear_sky_batch = X_future_clear_sky_batch.to(self.device)
                y_ghi_batch = y_ghi_batch.to(self.device)
                
                y_pred = model(X_batch, X_future_clear_sky_batch)
                
                y_pred_np = y_pred.cpu().numpy()
                y_true_np = y_ghi_batch.cpu().numpy()
                
                for step in range(self.num_future_frames):
                    y_pred_by_step[step].extend(y_pred_np[:, step])
                    y_true_by_step[step].extend(y_true_np[:, step])
        
        # Evaluate specific horizons
        reported_indices = [5, 11, 17, 23]  # 1hr, 2hr, 3hr, 4hr
        reported_names = ["1hr", "2hr", "3hr", "4hr"]
        combined_metrics = {}
        
        for i, step in enumerate(reported_indices):
            step_metrics = self.evaluate_model(
                np.array(y_true_by_step[step]), 
                np.array(y_pred_by_step[step])
            )
            
            print(f"\nMetrics for {step+1} steps ahead forecast:")
            for metric, value in step_metrics.items():
                print(f"{metric}: {value:.4f}")
                combined_metrics[f"{metric}_{reported_names[i]}"] = value
        
        return combined_metrics

    def optimize_hyperparameters(self, tensors: Dict[str, torch.Tensor], n_trials: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        storage_url = f"sqlite:///./database_{self.study_name}.db"
        
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=5, interval_steps=1
        )
        
        study = optuna.create_study(
            direction="minimize",
            pruner=pruner,
            study_name=self.study_name,
            storage=storage_url,
            load_if_exists=True,
        )
        
        study.optimize(
            lambda trial: self.objective(trial, tensors),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        return study.best_trial.params

    def train_final_model(self, dataset_name: str, embeddings_file: str, model_save_path: str):
        """Train the final forecasting model."""
        print("Loading data...")
        df, embeddings = self.load_data(dataset_name, embeddings_file)
        
        print("Generating timeseries data...")
        timeseries_data = self.generate_timeseries_data(df, embeddings)
        
        print("Preparing tensors...")
        tensors = self.prepare_tensors(timeseries_data, split_data=True)
        
        print("Optimizing hyperparameters...")
        best_params = self.optimize_hyperparameters(
            tensors, 
            n_trials=self.forecasting_config.get('n_trials', 50)
        )
        
        print(f"Best parameters: {best_params}")
        
        # Train final model with best parameters
        print("Training final model...")
        final_model = self.train_model(tensors, best_params)
        
        # Save final model
        torch.save(final_model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        
        # Save model configuration
        config_path = model_save_path.replace('.pth', '_config.json')
        model_config = {
            'best_params': best_params,
            'model_architecture': {
                'feature_dim': tensors["X_train"].shape[-1],
                'num_past_frames': self.num_past_frames,
                'num_future_frames': self.num_future_frames,
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"Model configuration saved to {config_path}")
        
        return final_model, best_params