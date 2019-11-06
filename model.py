import torch
import torch.nn as nn

class NeuralProcess(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, r_dim, s_dim, width=200):
        super(NeuralProcess, self).__init__()
        self.det_encoder = self.build_encoder(x_dim + y_dim, r_dim, width)
        self.stoch_encoder = self.build_encoder(x_dim + y_dim, r_dim, width)
        self.aggregator = self.build_aggregator()
        self.decoder = self.build_decoder(r_dim + z_dim + x_dim, y_dim, width)
        self.mu_layer = nn.Linear(s_dim, z_dim)
        self.logvar_layer = nn.Linear(s_dim, z_dim)

    def get_mu_logvar(self, data: torch.Tensor):
        s = self.stoch_encoder(data)
        s = self.aggregator(s)
        z_mu = self.mu_layer(s)
        z_logvar = self.logvar_layer(s)
        return z_mu, z_logvar

    def forward(self, data_context: torch.Tensor, x_target: torch.Tensor):
        r = self.det_encoder(data_context)
        r = self.aggregator(r)
        z_mu, z_logvar = self.get_mu_logvar(data_context)
        z = torch.randn(z_mu.size()) * torch.exp(0.5 * z_logvar) + z_mu
        num_target_points = x_target.shape[0]
        y = self.decoder(torch.cat((x_target.reshape(-1,1), z.repeat(num_target_points, 1), r.repeat(num_target_points, 1)), dim=1))
        return y

    def build_encoder(self, in_dim, out_dim, width):
        return self.build_simple_network(in_dim, width, out_dim)

    def build_aggregator(self):
        return self.mean_aggregator

    def mean_aggregator(self, x):
        return torch.mean(x, dim=0)

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

def NeuralProcessLoss(neural_process: NeuralProcess, data_context: torch.Tensor, data_target: torch.Tensor):
    x_target, y_target = data_target[:, 0], data_target[:, 1].reshape(-1, 1)
    mu_c, logvar_c = neural_process.get_mu_logvar(data_context)
    mu_t, logvar_t = neural_process.get_mu_logvar(data_target)
    d_kl = .5 * (
        torch.sum(torch.exp(logvar_t - logvar_c) - 1)
        + torch.dot((mu_t - mu_c) / torch.exp(logvar_c),(mu_t - mu_c))
        + (torch.sum(logvar_c) - torch.sum(logvar_t))
        )
    y_pred = neural_process(data_context, x_target)
    ll_fit = torch.nn.functional.mse_loss(y_target, y_pred)
    return ll_fit + d_kl
