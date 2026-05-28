"""
Hierarchical Spatial Prior Encoder (HSPE) for TGSD

Encodes topology-aware spatial priors over the complete electrode layout
by integrating local geometric relationships with region-level contextual information.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# EEG scalp regions for region-aware spatial fusion
# Based on standard 10-20 system regions
EEG_REGIONS = {
    'frontal_pole': list(range(0, 5)),       # Fp1, FpZ, Fp2, etc.
    'frontal': list(range(5, 14)),           # F3, F4, Fz, etc.
    'temporal_left': list(range(14, 20)),     # T7, TP7, etc.
    'temporal_right': list(range(20, 26)),    # T8, TP8, etc.
    'central': list(range(26, 36)),           # C3, C4, Cz, etc.
    'parietal': list(range(36, 46)),          # P3, P4, Pz, etc.
    'occipital': list(range(46, 62)),         # O1, O2, Oz, etc.
}


def get_region_indices(region_type='all'):
    """
    Get electrode indices for different scalp regions

    Args:
        region_type: 'all', 'frontal', 'temporal', 'central', 'parietal', 'occipital'

    Returns:
        List of electrode indices
    """
    if region_type == 'all':
        return list(range(62))
    elif region_type == 'frontal':
        return EEG_REGIONS['frontal_pole'] + EEG_REGIONS['frontal']
    elif region_type == 'temporal':
        return EEG_REGIONS['temporal_left'] + EEG_REGIONS['temporal_right']
    elif region_type == 'central':
        return EEG_REGIONS['central']
    elif region_type == 'parietal':
        return EEG_REGIONS['parietal']
    elif region_type == 'occipital':
        return EEG_REGIONS['occipital']
    else:
        return list(range(62))


def calculate_region_partition(n_channels=62):
    """
    Partition electrodes into regions for the hierarchical spatial prior

    For 62-channel EEG (10-20 system):
    - frontal pole: 5 channels
    - frontal: 9 channels
    - temporal left: 6 channels
    - temporal right: 6 channels
    - central: 10 channels
    - parietal: 10 channels
    - occipital: 16 channels
    """
    if n_channels == 62:
        return [5, 9, 6, 6, 10, 10, 16]  # 7 regions
    elif n_channels == 64:
        return [5, 9, 6, 6, 10, 10, 18]  # 7 regions for 64 channels
    else:
        # Adaptive partitioning
        per_region = n_channels // 7
        remainder = n_channels % 7
        partition = [per_region] * 7
        for i in range(remainder):
            partition[i] += 1
        return partition


class LocalTopologyPropagation(nn.Module):
    """
    Local topology propagation based on electrode adjacency
    Models local geometric relations using k-nearest neighbors graph
    """
    def __init__(self, in_features, hidden_features, k=8, tau=1.0):
        super().__init__()
        self.k = k
        self.tau = tau

        self.proj = nn.Linear(in_features, hidden_features)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, hidden_features))
        self.bias = nn.Parameter(torch.FloatTensor(hidden_features))

        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, h, adj_normalized):
        """
        Args:
            h: (B, K, F_h) - node features
            adj_normalized: (K, K) - normalized adjacency matrix

        Returns:
            h_loc: (B, K, F_h) - locally propagated features
        """
        # Linear transformation
        weighted_feature = torch.einsum('b i j, j d -> b i d', h, self.weight)
        # Graph convolution
        h_loc = torch.einsum('i j, b j d -> b i d', adj_normalized, weighted_feature) + self.bias
        h_loc = F.silu(h_loc)
        return h_loc


class RegionAwareFusion(nn.Module):
    """
    Region-aware contextual fusion
    Captures broader scalp organization using predefined brain regions
    """
    def __init__(self, hidden_features, num_heads=4, lambda_dist=0.1):
        super().__init__()
        self.hidden_features = hidden_features
        self.num_heads = num_heads
        self.lambda_dist = lambda_dist

        # Scoring function for attention
        self.psi = nn.Linear(hidden_features, hidden_features // num_heads)

        # Projection matrices for region contribution calculation
        self.W1 = nn.Linear(hidden_features, hidden_features // num_heads)
        self.W2 = nn.Linear(hidden_features, hidden_features // num_heads)

        # Region representation projection
        self.r_proj = nn.Linear(hidden_features, hidden_features)

    def forward(self, h_loc, positions, region_partition):
        """
        Args:
            h_loc: (B, K, F_h) - locally propagated features
            positions: (K, 3) - electrode 3D coordinates
            region_partition: list of region sizes

        Returns:
            h_reg: (B, K, F_h) - region-enhanced features
        """
        B, K, F_h = h_loc.shape
        device = h_loc.device

        # Partition into regions
        region_indices = []
        start_idx = 0
        for region_size in region_partition:
            region_indices.append(list(range(start_idx, start_idx + region_size)))
            start_idx += region_size

        # Compute regional features and positions
        num_regions = len(region_indices)
        r_q = []  # Regional representations
        c_q = []  # Regional coordinates

        for q, indices in enumerate(region_indices):
            region_features = h_loc[:, indices, :]  # (B, region_size, F_h)
            region_positions = positions[indices]  # (region_size, 3)

            # Attention weights within region
            scores = self.psi(region_features)  # (B, region_size, F_h//num_heads)
            att_weights = F.softmax(scores, dim=1)  # (B, region_size, F_h//num_heads)

            # Regional feature summary
            r_q.append(torch.sum(att_weights * region_features, dim=1).unsqueeze(1))  # (B, 1, F_h)

            # Regional position summary (center of mass)
            att_weights_2d = att_weights.mean(dim=-1, keepdim=True)  # (B, region_size, 1)
            c_q.append(torch.sum(att_weights_2d * region_positions.unsqueeze(0), dim=1).unsqueeze(1))  # (B, 1, 3)

        r_q = torch.cat(r_q, dim=1)  # (B, num_regions, F_h)
        c_q = torch.cat(c_q, dim=1)  # (B, num_regions, 3)

        # Compute contribution of each region to each channel
        alpha_iq = []  # (B, K, num_regions)
        for q in range(num_regions):
            # Feature similarity: (W1 * h_i_loc)^T * (W2 * r_q)
            h_i_w1 = torch.matmul(h_loc, self.W1.weight.T)  # (B, K, F_h//num_heads)
            r_q_w2 = torch.matmul(r_q[:, q:q+1], self.W2.weight.T)  # (B, 1, F_h//num_heads)

            feat_sim = torch.sum(h_i_w1 * r_q_w2, dim=-1)  # (B, K)

            # Spatial distance penalty
            pos_diff = torch.norm(positions.unsqueeze(1) - c_q[:, q:q+1], dim=-1)  # (B, K)
            dist_penalty = -self.lambda_dist * pos_diff

            # Combined score
            score = feat_sim + dist_penalty
            alpha_iq.append(F.softmax(score, dim=-1).unsqueeze(-1))  # (B, K, 1)

        alpha_iq = torch.cat(alpha_iq, dim=-1)  # (B, K, num_regions)

        # Compute region-enhanced representation
        h_reg = torch.bmm(alpha_iq, r_q)  # (B, K, F_h)
        h_reg = self.r_proj(h_reg)

        return h_reg


class HierarchicalSpatialPriorEncoder(nn.Module):
    """
    Hierarchical Spatial Prior Encoder (HSPE)

    Learns topology-aware priors over the complete electrode layout by:
    1. Initializing channel representations from EEG signals (observed) or learnable embeddings (target)
    2. Local topology propagation based on electrode adjacency
    3. Region-aware contextual fusion for broader scalp organization

    Args:
        n_channels: Total number of electrodes (K)
        in_feature_dim: Input EEG feature dimension (per channel)
        hidden_dim: Hidden dimension for representations
        k_neighbors: Number of nearest neighbors for local topology
        tau: Spatial decay parameter
        num_regions: Number of scalp regions
        dropout: Dropout rate
    """
    def __init__(
        self,
        n_channels: int,
        in_feature_dim: int,
        hidden_dim: int = 64,
        k_neighbors: int = 8,
        tau: float = 1.0,
        num_regions: int = 7,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.tau = tau

        # Learnable embeddings for target channels
        self.target_embeddings = nn.Parameter(torch.FloatTensor(n_channels, hidden_dim))
        nn.init.xavier_uniform_(self.target_embeddings)

        # Input projection for observed channels
        self.input_proj = nn.Linear(in_feature_dim, hidden_dim)

        # Local topology propagation
        self.local_gcn_1 = LocalTopologyPropagation(hidden_dim, hidden_dim, k=k_neighbors, tau=tau)
        self.local_gcn_2 = LocalTopologyPropagation(hidden_dim, hidden_dim, k=k_neighbors, tau=tau)

        # Region-aware fusion
        self.region_partition = calculate_region_partition(n_channels)
        self.region_fusion = RegionAwareFusion(hidden_dim, num_heads=4, lambda_dist=0.1)

        # Final projection
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Register positions (to be set externally)
        self.register_buffer('positions', torch.zeros(n_channels, 3))

    def set_positions(self, positions):
        """Set electrode 3D coordinates"""
        self.positions = positions

    def compute_local_adjacency(self, positions):
        """Compute normalized adjacency matrix based on k-nearest neighbors"""
        K = positions.shape[0]
        device = positions.device

        # Pairwise distances
        dist_matrix = torch.cdist(positions, positions)  # (K, K)

        # kNN adjacency (excluding self)
        _, indices = torch.topk(dist_matrix, k=self.k_neighbors + 1, largest=False)
        adjacency = torch.zeros(K, K, device=device)
        for i in range(K):
            neighbors = indices[i, 1:]  # exclude self
            adjacency[i, neighbors] = 1.0

        # Spatial decay
        dist_decay = torch.exp(-dist_matrix.pow(2) / (self.tau ** 2))
        adjacency = adjacency * dist_decay

        # Symmetric and normalize
        adjacency = (adjacency + adjacency.t()) / 2
        D = torch.sum(adjacency, dim=1)
        D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
        adj_normalized = torch.matmul(torch.matmul(D_inv_sqrt, adjacency), D_inv_sqrt)

        return adj_normalized

    def forward(self, x_obs, obs_mask, target_mask):
        """
        Forward pass

        Args:
            x_obs: (B, K_obs, L) - observed EEG signals
            obs_mask: (K,) - binary mask for observed channels (1 = observed)
            target_mask: (K,) - binary mask for target channels (1 = target)

        Returns:
            G_spa: (K, F_spa) - spatial prior for all channels
        """
        B = x_obs.shape[0]
        K = self.n_channels
        device = x_obs.device

        # Initialize channel representations
        # H^(0) = [phi(x_i) for observed] or [e_i for target]
        H = torch.zeros(B, K, self.hidden_dim, device=device)

        # Observed channels: project EEG sequence to hidden dimension
        obs_indices = torch.nonzero(obs_mask, as_tuple=True)[0]
        x_obs_mean = x_obs.mean(dim=-1)  # (B, K_obs) - temporal average
        x_obs_proj = F.silu(self.input_proj(x_obs_mean))  # (B, K_obs, hidden_dim)

        for idx, ch_idx in enumerate(obs_indices):
            H[:, ch_idx, :] = x_obs_proj[:, idx, :]

        # Target channels: use learnable embeddings
        tar_indices = torch.nonzero(target_mask, as_tuple=True)[0]
        tar_emb = self.target_embeddings[tar_indices]  # (K_tar, hidden_dim)
        H[:, tar_indices, :] = tar_emb.unsqueeze(0).expand(B, -1, -1)

        # Compute local adjacency
        adj_normalized = self.compute_local_adjacency(self.positions)

        # Local topology propagation
        H_loc = self.local_gcn_1(H, adj_normalized)
        H_loc = self.local_gcn_2(H_loc, adj_normalized)

        # Region-aware contextual fusion
        H_reg = self.region_fusion(H_loc, self.positions, self.region_partition)

        # Final spatial prior: combine local and regional
        H_final = H_loc + H_reg
        H_final = self.dropout(H_final)
        G_spa = self.proj(H_final)  # (B, K, hidden_dim)

        # Return spatial prior for all channels
        return G_spa


def build_hspe_from_positions(positions, config):
    """
    Build HSPE with electrode positions

    Args:
        positions: (K, 3) tensor of electrode 3D coordinates
        config: dict with model configuration

    Returns:
        HSPE model
    """
    model = HierarchicalSpatialPriorEncoder(
        n_channels=positions.shape[0],
        in_feature_dim=config.get('in_feature_dim', 1),
        hidden_dim=config.get('hidden_dim', 64),
        k_neighbors=config.get('k_neighbors', 8),
        tau=config.get('tau', 1.0),
        num_regions=config.get('num_regions', 7),
        dropout=config.get('dropout', 0.1)
    )
    model.set_positions(positions)
    return model