import sys
import os
import gzip
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import logging
from tqdm.auto import tqdm
import pandas as pd
from sklearn.decomposition import PCA


import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from sae import SparseAutoencoder
from utils import load_activations
from utils import prepare_residual_stream_data

layer_name='encoder.outer.residual2'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for transformer activations
    """
    def __init__(
        self, 
        input_dim, 
        hidden_dim=None,
        activation='relu',
        bias_decoder_init=None,
        tied_weights=False
    ):
        super().__init__()
        
        # Default hidden_dim to match input_dim if not specified
        if hidden_dim is None:
            hidden_dim = input_dim
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tied_weights = tied_weights
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Decoder (with optional tied weights)
        if tied_weights:
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
            
        # Initialize weights
        nn.init.xavier_normal_(self.encoder.weight)
        
        if not tied_weights:
            nn.init.xavier_normal_(self.decoder.weight)
            
        # Special initialization for decoder bias if specified
        if bias_decoder_init is not None and not tied_weights:
            nn.init.constant_(self.decoder.bias, bias_decoder_init)
            
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
    def encode(self, x):
        return self.activation(self.encoder(x))
    
    def decode(self, h):
        if self.tied_weights:
            return F.linear(h, self.encoder.weight.t(), self.decoder_bias)
        else:
            return self.decoder(h)
        
    def forward(self, x):
        h = self.encode(x)
        x_recon = self.decode(h)
        return x_recon, h

def l1_penalty(h, l1_coef):
    """L1 sparsity penalty"""
    return l1_coef * torch.mean(torch.abs(h))

def train_sae(
    model, 
    dataloader, 
    optimizer, 
    l1_coef=0.0,
    target_sparsity=None,
    kl_coef=0.0,
    num_epochs=10, 
    device="cuda" if torch.cuda.is_available() else "cpu",
    early_stopping_patience=5,
    scheduler=None,
    log_interval=10,
    validation_dataloader=None,
    dead_neuron_checker=None
):
    """
    Train the sparse autoencoder
    
    Args:
        model: The SAE model
        dataloader: Training data loader
        optimizer: PyTorch optimizer
        l1_coef: L1 regularization coefficient
        target_sparsity: Target activation probability (for KL divergence regularization)
        kl_coef: KL divergence regularization coefficient
        num_epochs: Number of training epochs
        device: Computation device
        early_stopping_patience: Number of epochs to wait before early stopping
        scheduler: Learning rate scheduler
        log_interval: Interval for logging
        validation_dataloader: Optional validation data loader
        dead_neuron_checker: Function to check and fix dead neurons
        
    Returns:
        dict: Training history
    """
    model.to(device)
    history = {
        'train_loss': [], 
        'recon_loss': [], 
        'l1_loss': [], 
        'kl_loss': [],
        'sparsity': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        recon_loss_sum = 0
        l1_loss_sum = 0
        kl_loss_sum = 0
        sparsity_sum = 0
        batch_count = 0
        
        for batch_idx, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            if isinstance(data, list) or isinstance(data, tuple):
                data = data[0]  # Handle cases where dataloader returns (data, target)
                
            data = data.to(device)
            optimizer.zero_grad()
            
            recon, hidden = model(data)
            
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(recon, data)
            
            # L1 sparsity penalty
            l1_loss = l1_penalty(hidden, l1_coef) if l1_coef > 0 else 0
            
            # KL divergence for target sparsity
            if target_sparsity is not None and kl_coef > 0:
                # Mean activation of hidden units across batch
                rho_hat = torch.mean(hidden, dim=0)
                # KL divergence between target_sparsity and rho_hat
                rho = torch.tensor(target_sparsity).to(device)
                kl_div = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
                kl_loss = kl_coef * torch.sum(kl_div)
            else:
                kl_loss = 0
            
            # Total loss
            loss = recon_loss + l1_loss + kl_loss
            
            loss.backward()
            optimizer.step()
            
            # Calculate sparsity
            sparsity = (hidden < 1e-4).float().mean().item()
            
            # Accumulate metrics
            train_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            l1_loss_sum += l1_loss.item() if isinstance(l1_loss, torch.Tensor) else l1_loss
            kl_loss_sum += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            sparsity_sum += sparsity
            batch_count += 1
            
            if (batch_idx + 1) % log_interval == 0:
                logger.debug(f"Batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item():.4f}")
                
        # Check for dead neurons and fix if needed
        if dead_neuron_checker is not None:
            dead_neuron_checker(model, dataloader, device)
        
        # Calculate epoch averages
        avg_train_loss = train_loss / batch_count
        avg_recon_loss = recon_loss_sum / batch_count
        avg_l1_loss = l1_loss_sum / batch_count
        avg_kl_loss = kl_loss_sum / batch_count
        avg_sparsity = sparsity_sum / batch_count
        
        # Validation
        val_loss = 0
        if validation_dataloader is not None:
            model.eval()
            val_batch_count = 0
            with torch.no_grad():
                for val_data in validation_dataloader:
                    if isinstance(val_data, list) or isinstance(val_data, tuple):
                        val_data = val_data[0]
                    val_data = val_data.to(device)
                    val_recon, val_hidden = model(val_data)
                    val_recon_loss = F.mse_loss(val_recon, val_data)
                    val_l1_loss = l1_penalty(val_hidden, l1_coef) if l1_coef > 0 else 0
                    
                    # KL divergence
                    if target_sparsity is not None and kl_coef > 0:
                        val_rho_hat = torch.mean(val_hidden, dim=0)
                        val_rho = torch.tensor(target_sparsity).to(device)
                        val_kl_div = val_rho * torch.log(val_rho / val_rho_hat) + (1 - val_rho) * torch.log((1 - val_rho) / (1 - val_rho_hat))
                        val_kl_loss = kl_coef * torch.sum(val_kl_div)
                    else:
                        val_kl_loss = 0
                        
                    val_total_loss = val_recon_loss + val_l1_loss + val_kl_loss
                    val_loss += val_total_loss.item()
                    val_batch_count += 1
                    
            val_loss /= val_batch_count
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Learning rate scheduler step
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_train_loss)
            else:
                scheduler.step()
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['l1_loss'].append(avg_l1_loss)
        history['kl_loss'].append(avg_kl_loss)
        history['sparsity'].append(avg_sparsity)
        history['val_loss'].append(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_train_loss:.6f}, Recon = {avg_recon_loss:.6f}, " +
                   f"L1 = {avg_l1_loss:.6f}, KL = {avg_kl_loss:.6f}, Sparsity = {avg_sparsity:.4f}, Val = {val_loss:.6f}")
    
    return history

def fix_dead_neurons(model, dataloader, device, activation_threshold=0.01, fix_method='reinit'):
    """
    Check for and fix dead neurons during training
    
    Args:
        model: SAE model
        dataloader: Data loader for activation check
        device: Computation device
        activation_threshold: Threshold below which a neuron is considered dead
        fix_method: Method to fix dead neurons ('reinit', 'clone', or 'noise')
    """
    model.eval()
    
    # Create a tensor to track cumulative activations
    neuron_activations = torch.zeros(model.hidden_dim).to(device)
    total_samples = 0
    
    # Sample some batches to check for dead neurons
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= 10:  # Only check a few batches for efficiency
                break
                
            if isinstance(data, list) or isinstance(data, tuple):
                data = data[0]
            
            data = data.to(device)
            _, hidden = model(data)
            
            # Track if neurons activate
            neuron_activations += (hidden > activation_threshold).float().sum(dim=0)
            total_samples += data.size(0)
    
    # Calculate activation frequency
    activation_freq = neuron_activations / total_samples
    
    # Identify dead neurons
    dead_neurons = (activation_freq < 0.01).nonzero().view(-1)
    
    if len(dead_neurons) > 0:
        logger.info(f"Found {len(dead_neurons)} dead neurons. Fixing...")
        
        if fix_method == 'reinit':
            # Reinitialize weights for dead neurons
            with torch.no_grad():
                model.encoder.weight.data[dead_neurons] = torch.randn_like(model.encoder.weight.data[dead_neurons]) * 0.01
                
                if not model.tied_weights:
                    # Fix: Handle each dead neuron individually for decoder
                    for dead_idx in dead_neurons:
                        model.decoder.weight.data[:, dead_idx] = torch.randn_like(model.decoder.weight.data[:, dead_idx]) * 0.01
                    
        elif fix_method == 'clone':
            # Clone active neurons to replace dead ones
            active_neurons = (activation_freq >= 0.01).nonzero().view(-1)
            
            if len(active_neurons) > 0:
                with torch.no_grad():
                    for dead_idx in dead_neurons:
                        # Use a random active neuron as template
                        active_idx = active_neurons[torch.randint(0, len(active_neurons), (1,))].item()
                        
                        # Copy with small noise
                        model.encoder.weight.data[dead_idx] = model.encoder.weight.data[active_idx] + torch.randn_like(model.encoder.weight.data[dead_idx]) * 0.01
                        
                        if not model.tied_weights:
                            # Fix: Handle each index as a scalar to avoid shape mismatch
                            noise = torch.randn_like(model.decoder.weight.data[:, dead_idx]) * 0.01
                            model.decoder.weight.data[:, dead_idx] = model.decoder.weight.data[:, active_idx] + noise
            else:
                # If no active neurons, fall back to random init
                logger.warning("No active neurons found. Using random initialization instead.")
                fix_dead_neurons(model, dataloader, device, activation_threshold, 'reinit')
                
        elif fix_method == 'noise':
            # Add noise to the dead neuron weights
            with torch.no_grad():
                model.encoder.weight.data[dead_neurons] += torch.randn_like(model.encoder.weight.data[dead_neurons]) * 0.1
                
                if not model.tied_weights:
                    # Fix: Handle each dead neuron individually for decoder
                    for dead_idx in dead_neurons:
                        model.decoder.weight.data[:, dead_idx] += torch.randn_like(model.decoder.weight.data[:, dead_idx]) * 0.1
        else:
            raise ValueError(f"Unknown fix method: {fix_method}")

def evaluate_sae(model, dataloader, device):
    """
    Evaluate an SAE model comprehensively
    
    Args:
        model: Trained SAE model
        dataloader: Data loader for evaluation
        device: Computation device
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    model.to(device)
    
    reconstruction_error = 0
    all_hidden = []
    all_inputs = []
    batch_count = 0
    
    with torch.no_grad():
        for data in dataloader:
            if isinstance(data, list) or isinstance(data, tuple):
                data = data[0]
                
            data = data.to(device)
            recon, hidden = model(data)
            
            # Collect hidden representations
            all_hidden.append(hidden.cpu().numpy())
            all_inputs.append(data.cpu().numpy())
            
            # Calculate reconstruction error
            batch_recon_error = F.mse_loss(recon, data).item()
            reconstruction_error += batch_recon_error
            batch_count += 1
    
    # Average reconstruction error
    avg_recon_error = reconstruction_error / batch_count
    
    # Concatenate all hidden representations and inputs
    all_hidden = np.vstack(all_hidden)
    all_inputs = np.vstack(all_inputs)
    
    # Sparsity metrics
    l0_sparsity = (np.abs(all_hidden) < 1e-4).mean()
    l1_norm = np.mean(np.abs(all_hidden), axis=1).mean()
    
    # Feature activation statistics
    per_neuron_activation = (np.abs(all_hidden) >= 1e-4).mean(axis=0)
    
    # Calculate Hoyer sparsity
    l1_norms = np.sum(np.abs(all_hidden), axis=1)
    l2_norms = np.sqrt(np.sum(all_hidden**2, axis=1))
    n_features = all_hidden.shape[1]
    hoyer_values = []
    
    for i in range(len(l1_norms)):
        if l2_norms[i] > 0:
            hoyer = (np.sqrt(n_features) - l1_norms[i] / l2_norms[i]) / (np.sqrt(n_features) - 1)
        else:
            hoyer = 1.0
        hoyer_values.append(hoyer)
    
    hoyer_sparsity = np.mean(hoyer_values)
    
    # Feature diversity: compute cosine similarity between features
    encoder_weights = model.encoder.weight.data.cpu().numpy()
    feature_similarities = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(i, n_features):
            norm_i = np.linalg.norm(encoder_weights[i])
            norm_j = np.linalg.norm(encoder_weights[j])
            
            if norm_i > 0 and norm_j > 0:
                sim = np.abs(np.dot(encoder_weights[i], encoder_weights[j]) / (norm_i * norm_j))
            else:
                sim = 0
                
            feature_similarities[i, j] = sim
            feature_similarities[j, i] = sim
    
    # Average off-diagonal cosine similarity (measure of redundancy)
    mask = ~np.eye(n_features, dtype=bool)
    avg_cos_sim = feature_similarities[mask].mean()
    
    # PCA on encoder weights to measure intrinsic dimensionality
    pca = PCA()
    pca.fit(encoder_weights)
    var_explained_ratio = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(var_explained_ratio)
    intrinsic_dim_90 = np.sum(cumulative_var < 0.9) + 1  # Number of components for 90% variance
    
    # Results
    metrics = {
        'reconstruction_error': avg_recon_error,
        'l0_sparsity': l0_sparsity,
        'l1_norm': l1_norm,
        'hoyer_sparsity': hoyer_sparsity,
        'avg_cos_similarity': avg_cos_sim,
        'intrinsic_dim_90': intrinsic_dim_90,
        'active_neuron_ratio': (per_neuron_activation > 0.01).mean(),
        'dead_neuron_ratio': (per_neuron_activation < 0.01).mean(),
        'mean_neuron_activation': per_neuron_activation.mean(),
        'std_neuron_activation': per_neuron_activation.std(),
    }
    
    return metrics

def objective(trial, data_tensor, input_dim, batch_size=128, num_epochs=30, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Optuna objective function for hyperparameter search
    
    Args:
        trial: Optuna trial
        data_tensor: Training data tensor
        input_dim: Input dimension
        batch_size: Batch size
        num_epochs: Maximum number of epochs
        device: Computation device
        
    Returns:
        float: Objective score (lower is better)
    """
    # Get hyperparameters from the trial
    hidden_dim_factor = trial.suggest_float("hidden_dim_factor", 2.0, 5.0)
    hidden_dim = int(input_dim * hidden_dim_factor)
    
    l1_coef = trial.suggest_float("l1_coef", 1e-5, 1e-2, log=True)
    
    target_sparsity = trial.suggest_float("target_sparsity", 0.01, 0.1)
    kl_coef = trial.suggest_float("kl_coef", 0.0, 1.0)
    
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "gelu"])
    tied_weights = trial.suggest_categorical("tied_weights", [True, False])
    
    # Create dataset and dataloader
    dataset = TensorDataset(data_tensor)
    
    # Split data for validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = SparseAutoencoder(
        input_dim=input_dim, 
        hidden_dim=hidden_dim,
        activation=activation,
        tied_weights=tied_weights
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Dead neuron handler
    dead_neuron_handler = lambda m, dl, d: fix_dead_neurons(
        m, dl, d, 
        activation_threshold=0.01, 
        fix_method='clone'
    )
    
    # Train the model
    history = train_sae(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        l1_coef=l1_coef,
        target_sparsity=target_sparsity,
        kl_coef=kl_coef,
        num_epochs=num_epochs,
        device=device,
        early_stopping_patience=5,
        scheduler=scheduler,
        log_interval=20,
        validation_dataloader=val_loader,
        dead_neuron_checker=dead_neuron_handler
    )
    
    # Evaluate the model
    metrics = evaluate_sae(model, val_loader, device)
    
    # Multi-objective optimization: we want high sparsity but low reconstruction error
    # Define a combined score
    recon_error = metrics['reconstruction_error']
    sparsity = metrics['hoyer_sparsity']
    
    # We want high sparsity (close to 1) and low reconstruction error
    # Balance between reconstruction quality and sparsity
    score = recon_error * (2.0 - sparsity)
    
    # Report metrics
    trial.set_user_attr('hoyer_sparsity', sparsity)
    trial.set_user_attr('reconstruction_error', recon_error)
    trial.set_user_attr('dead_neuron_ratio', metrics['dead_neuron_ratio'])
    trial.set_user_attr('intrinsic_dim_90', metrics['intrinsic_dim_90'])
    trial.set_user_attr('l0_sparsity', metrics['l0_sparsity'])
    
    return score

def run_hyperparameter_search(
    data_tensor, 
    input_dim,
    n_trials=100,
    study_name="sae_optimization",
    storage=None,
    batch_size=128,
    num_epochs=30,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Run hyperparameter search for SAE
    
    Args:
        data_tensor: Input data tensor
        input_dim: Input dimension
        n_trials: Number of trials for optimization
        study_name: Study name for Optuna
        storage: Storage URL for Optuna
        batch_size: Batch size
        num_epochs: Maximum epochs per trial
        device: Computation device
        
    Returns:
        study: Optuna study object
    """
    logger.info(f"Starting hyperparameter search with {n_trials} trials")
    
    # Create the study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True
    )
    
    # Run optimization
    objective_func = lambda trial: objective(
        trial, data_tensor, input_dim, batch_size, num_epochs, device
    )
    
    study.optimize(objective_func, n_trials=n_trials)
    
    # Print results to both log and stdout
    logger.info("Best hyperparameters:")
    logger.info(study.best_params)
    logger.info(f"Best score: {study.best_value}")
    
    # Add explicit print statements for stdout
    print("\n" + "="*50)
    print("BEST HYPERPARAMETERS:")
    print("="*50)
    
    for param_name, param_value in study.best_params.items():
        print(f"{param_name}: {param_value}")
    
    print(f"\nBest score: {study.best_value:.6f}")
    print("\nBest trial metrics:")
    
    for key in ['hoyer_sparsity', 'reconstruction_error', 'dead_neuron_ratio', 'intrinsic_dim_90', 'l0_sparsity']:
        value = study.best_trial.user_attrs[key]
        logger.info(f"Best trial {key}: {value}")
        print(f"  {key}: {value:.6f}")
    
    # Visualization
    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)
        
        figs = {
            'optimization_history': fig1,
            'param_importances': fig2
        }
        return study, figs
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")
        return study, None

def train_sae_with_best_params(
    study, 
    data_tensor, 
    input_dim,
    batch_size=128,
    num_epochs=100,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train SAE with the best hyperparameters found
    
    Args:
        study: Optuna study from hyperparameter search
        data_tensor: Input data tensor
        input_dim: Input dimension
        batch_size: Batch size
        num_epochs: Maximum number of epochs
        device: Computation device
        
    Returns:
        model: Trained model
        history: Training history
        metrics: Evaluation metrics
    """
    # Get best hyperparameters
    params = study.best_params
    
    # Calculate hidden dimension
    hidden_dim = int(input_dim * params['hidden_dim_factor'])
    
    # Create dataset and dataloader
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = SparseAutoencoder(
        input_dim=input_dim, 
        hidden_dim=hidden_dim,
        activation=params['activation'],
        tied_weights=params['tied_weights']
    )
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=params['lr'], 
        weight_decay=params['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Train the model
    history = train_sae(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        l1_coef=params['l1_coef'],
        target_sparsity=params['target_sparsity'],
        kl_coef=params['kl_coef'],
        num_epochs=num_epochs,
        device=device,
        early_stopping_patience=10,
        scheduler=scheduler,
        log_interval=20,
        validation_dataloader=None,
        dead_neuron_checker=lambda m, dl, d: fix_dead_neurons(m, dl, d, fix_method='clone')
    )
    
    # Evaluate the model
    metrics = evaluate_sae(model, dataloader, device)
    
    return model, history, metrics

if __name__ == "__main__":
    # Example usage

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"sae.{layer_name}_{timestamp}"
    #os.makedirs(run_dir, exist_ok=True)
    #checkpoint_dir = os.path.join(run_dir, "checkpoints")
    #os.makedirs(checkpoint_dir, exist_ok=True)
    
    filepath = '../data/activations/random_solutions_activations_10k.pkl.gz'
    collected_act = load_activations(filepath)

    # Prepare the training data - now handles variable sequence lengths
    data = prepare_residual_stream_data(collected_act,site_name='residual_stream',layer_name=layer_name)
    
    # 1. Prepare data (example for MLP activations from transformer)
    # This would typically come from your transformer model
    input_dim = 256  # Example dimension
    n_samples = 100000
    
    # Generate synthetic data (replace with your actual transformer activations)
    #np.random.seed(42)
    data = np.random.randn(n_samples, input_dim)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    # Display information about the search range
    print("\nRunning hyperparameter search with:")
    print(f"- Input dimension: {input_dim}")
    print(f"- Hidden dimension range: {int(input_dim * 2.0)} to {int(input_dim * 5.0)} (2x to 5x)")
    print(f"- Number of samples: {n_samples}")
    print(f"- Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # 2. Run hyperparameter search
    study, figs = run_hyperparameter_search(
        data_tensor=data_tensor,
        input_dim=input_dim,
        n_trials=50,  # Increase for better results
        batch_size=256,
        num_epochs=20
    )
    
    # Print a summary of the top 5 trials
    print("\n" + "="*50)
    print("TOP 5 TRIALS:")
    print("="*50)
    
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))[:5]
    
    for i, trial in enumerate(top_trials):
        if trial.value is None:
            continue
            
        print(f"\nRank {i+1} (Trial {trial.number}):")
        print(f"  Score: {trial.value:.6f}")
        
        for param_name, param_value in trial.params.items():
            print(f"  {param_name}: {param_value}")
            
        # Print select metrics
        for key in ['hoyer_sparsity', 'reconstruction_error', 'l0_sparsity']:
            if key in trial.user_attrs:
                print(f"  {key}: {trial.user_attrs[key]:.6f}")
    
    # 3. Train with best parameters
    print("\n" + "="*50)
    print("TRAINING WITH BEST PARAMETERS")
    print("="*50)
    
    model, history, metrics = train_sae_with_best_params(
        study=study,
        data_tensor=data_tensor,
        input_dim=input_dim,
        batch_size=256,
        num_epochs=50
    )
    
    # 4. Print final results
    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS")
    print("="*50)
    
    # Organize metrics into categories
    metric_categories = {
        "Sparsity Metrics": ["l0_sparsity", "hoyer_sparsity", "gini_coefficient"],
        "Reconstruction Quality": ["reconstruction_error"],
        "Feature Analysis": ["avg_cos_similarity", "intrinsic_dim_90"],
        "Neuron Health": ["active_neuron_ratio", "dead_neuron_ratio", "mean_neuron_activation"]
    }
    
    for category, metric_keys in metric_categories.items():
        print(f"\n{category}:")
        for key in metric_keys:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.6f}")
        
    # 5. Save the model and best parameters
    torch.save({
        "model_state_dict": model.state_dict(),
        "hyperparameters": study.best_params,
        "metrics": metrics
    }, "best_sae_model.pt")
    
    print(f"\nModel saved to best_sae_model.pt")
    
    # 6. Plot training history
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Total Loss')
    plt.plot(history['recon_loss'], label='Reconstruction Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['sparsity'], label='Sparsity')
    plt.title('Sparsity Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Sparsity (% of zeros)')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history['l1_loss'], label='L1 Loss')
    plt.plot(history['kl_loss'], label='KL Loss')
    plt.title('Regularization Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.semilogy(history['train_loss'], label='Log Loss')
    plt.title('Log Scale Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('sae_training_history.png')
    print(f"Training history plot saved to sae_training_history.png")
    plt.show()