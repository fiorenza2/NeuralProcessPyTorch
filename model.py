import torch
import torch.nn as nn

class NeuralProcess(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, r_dim, s_dim, width=200):
        super(NeuralProcess, self).__init__()
        self.det_encoder = self.build_encoder(x_dim+y_dim, r_dim, width)
        self.stoch_encoder = self.build_encoder(x_dim+y_dim, r_dim, width)
        self.aggregator = self.build_aggregator()
        self.decoder = self.build_decoder(r_dim+z_dim+x_dim, y_dim, width)
        self.mu_layer = nn.Linear(s_dim, z_dim)
        self.sigma_layer = nn.Linear(s_dim, z_dim)

    def forward(self, x_context: torch.Tensor, x_target: torch.Tensor):
        r = self.det_encoder(x_context)
        r = self.aggregator(r)
        s = self.stoch_encoder(x_context)
        s = self.aggregator(s)
        z_mu = self.mu_layer(s)
        z_sigma = self.sigma_layer(s)
        z = torch.randn(s.size) * z_sigma + z_mu
        num_target_points = x_target.shape[0]
        y = self.decoder(torch.cat((x_target, z.repeat(num_target_points), r.repeat(num_target_points))))
        return y

    def build_encoder(self, in_dim, out_dim, width):
        return self.build_simple_network(in_dim, width, out_dim)

    def build_aggregator(self):
        return torch.mean

    def build_decoder(self, latent_dim, out_dim, width):
        return self.build_simple_network(latent_dim, width, out_dim)
        
    def build_simple_network(self, in_dim, hid_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

def NeuralProcessLoss(y_pred, y_actual):
    