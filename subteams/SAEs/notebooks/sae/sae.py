import torch
import torch.nn as nn
from lightning.pytorch import LightningModule


class SparseAutoencoder(LightningModule):
    def __init__(self, input_dim=256, latent_dim=1024, name=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.name = name

        self.save_hyperparameters()
        
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.encoder.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.decoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
    
    def encode(self, input):
        return torch.relu(self.encoder(input))
    
    def decode(self, lattent):
        return self.decoder(lattent)
    
    def forward(self, input):
        lattent = self.encode(input)
        return self.decode(lattent), lattent