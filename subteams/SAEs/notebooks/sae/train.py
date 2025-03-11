import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
from tqdm import tqdm
import os
from datetime import datetime

from torch import optim
import torch.nn as nn
from tqdm import tqdm
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from sae import SparseAutoencoder

from utils import load_activations
from utils import prepare_residual_stream_data


def train_sae(model, data_loader, num_epochs=100, learning_rate=1e-3, l1_lambda=1e-3,
              device="cuda", checkpoint_dir="checkpoints", patience=10, min_delta=1e-4):
    print(f"Started training using device: {device}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()
    
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath=checkpoint_dir,
        filename='sae-epoch{epoch:02d}-train_loss{train_loss:.4f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='train_loss',
        min_delta=min_delta,
        patience=patience,
        verbose=True,
        mode='min'
    )
    
    best_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    total_epochs = 0
    
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in data_loader:
            x = batch[0].to(device)
            
            x_reconstructed,h = model(x)
            
            loss = mse_loss(x_reconstructed, x) + l1_lambda * torch.mean(torch.abs(h))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # Save best model
            checkpoint_path = os.path.join(checkpoint_dir, f"sae_best_{model.name}.pt")
            torch.save(model.state_dict(), checkpoint_path)
        
        if (epoch + 1) % 10 == 0:
            print(f"[Train] Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")
            checkpoint_path = os.path.join(checkpoint_dir, f"sae_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)

        current = torch.tensor(avg_loss, device=device)
        
        if early_stopping.monitor_op(current - early_stopping.min_delta, early_stopping.best_score):
            early_stopping.best_score = current
            early_stopping.wait_count = 0
            print(f"[Early stopping] Epoch:{epoch} Wait count reset (best: {early_stopping.best_score:.6f})")
        else:
            early_stopping.wait_count += 1
            print(f"[Early stopping] Epoch:{epoch} Wait count: {early_stopping.wait_count}/{early_stopping.patience}")
            
            if early_stopping.wait_count >= early_stopping.patience:
                early_stopping.stopped_epoch = epoch
                print(f"[Early stopping] Triggered after {epoch} epochs")
                break
        
        total_epochs = epoch + 1
    
    best_model_path = os.path.join(checkpoint_dir, "sae_best.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path,weights_only=True))
        print(f"[Train] Loaded best model from epoch {best_epoch+1} with loss {best_loss:.6f}")
    
    final_path = os.path.join(checkpoint_dir, "sae_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"[Train] Final model saved to {final_path}")
    
    return model, best_loss