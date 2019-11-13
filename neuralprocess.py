import torch
import torch.nn as nn
from .models import *


class NeuralProcess(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, z_dim: int, r_dim: int, s_dim: int, width: int = 200):
        super(NeuralProcess, self).__init__()
        self.determinstic_encoder = DeterministicEncoder(x_dim, y_dim, r_dim)
        self.latent_encoder = LatentEncoder(x_dim, y_dim, z_dim)
        self.decoder = Decoder(x_dim, z_dim, r_dim, y_dim)

    def forward(self, x_context: torch.Tensor, y_context: torch.Tensor, x_target: torch.Tensor):
        r = self.determinstic_encoder(x_context, y_context)
        z_mu, z_sigma = self.latent_encoder(x_context, y_context)
        z = torch.randn(z_mu.size()) * z_sigma + z_mu
        num_target_points = x_target.shape[0]
        y = self.decoder(torch.cat((x_target.reshape(-1,1), z.repeat(num_target_points, 1), r.repeat(num_target_points, 1)), dim=1))
        return y
