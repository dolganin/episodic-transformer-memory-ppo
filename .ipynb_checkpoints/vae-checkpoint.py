# vae.py
import torch, torch.nn as nn, torch.nn.functional as F

class MemoryVAE(nn.Module):
    def __init__(self, feature_dim: int, latent_dim: int = 64, hidden: int = 256):
        super().__init__()
        # encoder
        self.enc = nn.Sequential(
            nn.LayerNorm(feature_dim),                 # <-- нормализация
            nn.Linear(feature_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mu      = nn.Linear(hidden, latent_dim)
        self.log_var = nn.Linear(hidden, latent_dim)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, feature_dim)
        )

    def encode(self, x):
        h = self.enc(x)
        mu, log_var = self.mu(h), self.log_var(h)
        log_var = torch.clamp(log_var, -10., 10.)       # <-- clamp
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z   = self.reparameterize(mu, log_var)
        x_hat = self.dec(z)
        return x_hat, mu, log_var

