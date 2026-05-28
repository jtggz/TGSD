import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np
from typing import Optional


class RMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, d_model: int, eps: float=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    """Conv1d with Kaiming initialization"""
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    layer = nn.utils.weight_norm(layer)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def flip(seq):
    """Flip sequence along dimension 2"""
    return torch.flip(seq, [2])


def swish(x):
    """Swish activation"""
    return x * torch.sigmoid(x)


def cal_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """Embed diffusion steps into higher dimensional space"""
    assert diffusion_step_embed_dim_in % 2 == 0
    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).to(diffusion_steps.device)
    _embed = diffusion_steps.float() * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed), torch.cos(_embed)), 1)
    return diffusion_step_embed


class DiffusionEmbedding(nn.Module):
    """Embedding for diffusion timestep"""
    def __init__(self, num_steps, embedding_dim, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            'embedding',
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.proj1 = nn.Linear(embedding_dim, projection_dim)
        self.proj2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.proj1(x)
        x = F.silu(x)
        x = self.proj2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


def normalize_adj(adj):
    """Normalize adjacency matrix"""
    D = torch.diag(torch.sum(adj, dim=1))
    D_ = torch.diag(torch.diag(1 / torch.sqrt(D)))
    lap_matrix = torch.matmul(D_, torch.matmul(adj, D_))
    return lap_matrix


def calculate_local_adjacency(positions, k=8, tau=1.0):
    """
    Calculate local adjacency matrix based on electrode positions

    Args:
        positions: electrode coordinates (K, 3)
        k: number of nearest neighbors
        tau: spatial decay parameter

    Returns:
        adjacency matrix (K, K)
    """
    K = positions.shape[0]
    device = positions.device

    # Calculate pairwise Euclidean distances
    dist_matrix = torch.cdist(positions, positions)  # (K, K)

    # Get k nearest neighbors for each electrode
    _, indices = torch.topk(dist_matrix, k=k + 1, largest=False)  # include self

    # Create adjacency matrix based on kNN
    adjacency = torch.zeros(K, K, device=device)
    for i in range(K):
        neighbors = indices[i, 1:]  # exclude self
        adjacency[i, neighbors] = 1.0

    # Apply spatial decay: A_ij = exp(-d_ij^2 / tau^2) if j in N_k(i)
    dist_decay = torch.exp(-dist_matrix.pow(2) / (tau ** 2))
    adjacency = adjacency * dist_decay

    # Make symmetric
    adjacency = (adjacency + adjacency.t()) / 2

    return adjacency


def get_spatial_decay_matrix(positions, tau=1.0):
    """
    Calculate spatial decay matrix based on electrode positions

    Args:
        positions: electrode coordinates (K, 3)
        tau: spatial decay parameter

    Returns:
        spatial decay matrix (K, K)
    """
    dist_matrix = torch.cdist(positions, positions)  # (K, K)
    return torch.exp(-dist_matrix.pow(2) / (tau ** 2))