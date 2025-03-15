import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from torch.utils.data import Dataset
from torch.autograd import Variable
from vendi import score

# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# Squared Error Function
def squared_error(x, y):
    return ((x - y) ** 2).mean()

# Swish Activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Self-Attention Block
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        return self.norm(x + self.dropout(attn_output))

# Self-Attention Time-Series Model
class SelfAttentionTimeSeries(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads, attention, num_layers, device):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.attention = attention
        self.attention_layers = nn.ModuleList(
            [SelfAttention(output_dim, num_heads) for _ in range(num_layers)]
        )
        self.enc1 = EncoderBlock(input_dim + d_model, input_dim)
        self.dec3 = EncoderBlock(input_dim * 2 + d_model, output_dim)

    def forward(self, x, ts):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        embedded_k = get_timestep_embedding(ts, self.d_model).repeat(x.shape[1], 1).unsqueeze(0)
        x_in = self.enc1(torch.cat((x, embedded_k), -1))
        x_in = self.dec3(torch.cat((x, x_in, embedded_k), -1))
        if self.attention:
            for attn_layer in self.attention_layers:
                x_in = attn_layer(x_in)
        return x_in

# Timestep Embedding
def get_timestep_embedding(timesteps, embedding_dim):
    timesteps = torch.tensor([timesteps])
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -math.log(10000) / (half_dim - 1))
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return F.pad(emb, (0, 1, 0, 0)) if embedding_dim % 2 == 1 else emb

# Alpha Alternative Model
class AlphaAlt(nn.Module):
    def __init__(self, latent_dim, obser_dim, sigma_x, sigma_z, vendi_L, vendi_q, importance_sample_size, n_layers, device):
        super().__init__()
        self.device = device
        self.latent_dim = torch.tensor([latent_dim], requires_grad=False).to(device)
        self.obser_dim = torch.tensor([obser_dim], requires_grad=False).to(device)
        self.sigma_x = torch.tensor([sigma_x], requires_grad=False).to(device)
        self.sigma_z = torch.tensor([sigma_z], requires_grad=False).to(device)
        self.importance_sample_size = importance_sample_size
        self.n_layers = n_layers
        self.vendi_L = vendi_L
        self.vendi_q = vendi_q
        self.alpha_encoder = nn.Linear(1, 1)
        self.g_theta = SelfAttentionTimeSeries(latent_dim, obser_dim, 10, 1, False, n_layers, device)
        self.f_phi_x = SelfAttentionTimeSeries(obser_dim, latent_dim, 10, 1, False, n_layers, device)

    def forward(self, obsrv, mask, eps_x, eps_z, obsr_enable):
        batch_size, seq_len = obsrv.shape[1], obsrv.shape[0]
        self.z_hat = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
        self.x_hat = torch.zeros(seq_len, batch_size, self.obser_dim).to(self.device)
        self.alpha = torch.zeros(seq_len, batch_size, 1).to(self.device)
        self.z_hat[0] = self.sigma_x * Variable(torch.randn(self.z_hat[0].shape)).to(self.device)
        vs = torch.zeros(batch_size, 1).to(self.device)
        self.alpha[0] = torch.sigmoid(self.alpha_encoder(vs)) * (1 - self.sigma_z ** 2 - 1e-4)

        for k in range(1, seq_len):
            eps_z = (eps_z + self.sigma_z * torch.randn(self.z_hat[0].shape).to(self.device)) / 2
            eps_x = (eps_x + self.sigma_x * torch.randn(self.x_hat[0].shape).to(self.device)) / 2
            vs = score(self.z_hat[max(0, k - self.vendi_L):k], squared_error, q=self.vendi_q) * torch.ones(batch_size, 1).to(self.device)
            self.alpha[k] = torch.sigmoid(self.alpha_encoder(vs)) * (1 - self.sigma_z ** 2 - 1e-4)

            self.x_hat[k] = torch.sqrt(1 - self.sigma_x ** 2) * self.g_theta(self.z_hat[k - 1:k].clone(), k) + eps_x
            self.z_hat[k] = torch.sqrt(self.alpha[k]) * self.f_phi_x(self.x_hat[k], k) + torch.sqrt(1 - self.alpha[k] - self.sigma_z ** 2) * self.z_hat[k - 1].clone() + eps_z

        return self.z_hat

    def loss(self, a, b, c, z):
        L1 = F.mse_loss(torch.sqrt(1 - self.sigma_x ** 2) * self.x_hat, self.obsrv) * self.sigma_z ** 2
        L2 = F.mse_loss(z[1:], torch.sqrt(1 - self.alpha[:-1] - self.sigma_z ** 2) * self.z_hat[:-1] + torch.sqrt(self.alpha[1:]) * self.z_hat[1:]) * self.sigma_x ** 2
        return b * L2 + a * L1

# Dataset Classes
class GetDataset(Dataset):
    def __init__(self, x, z, device):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.z = torch.tensor(z, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.z[index]

# Utility Functions
def init_weights(m):
    for param in m.parameters():
        nn.init.uniform_(param.data, -0.05, 0.05)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
