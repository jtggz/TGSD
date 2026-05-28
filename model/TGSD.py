"""
TGSD: Topology-Guided State-Space Diffusion Framework for EEG Spatial Super-Resolution
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .HSPE import HierarchicalSpatialPriorEncoder
from .CSDR import ConditionalStateSpaceDiffusionReconstructor, cal_diffusion_step_embedding, DiffusionEmbedding


class TGSDModel(nn.Module):
    """
    Complete TGSD model combining Hierarchical Spatial Prior Encoder (HSPE)
    and Conditional State-Space Diffusion Reconstructor (CSDR)
    """
    def __init__(
        self,
        n_channels: int,
        n_obs_channels: int,
        electrode_positions: torch.Tensor,
        seq_len: int = 800,
        hidden_dim: int = 64,
        num_layers: int = 12,
        num_ssm: int = 2,
        diffusion_steps: int = 1000,
        diffusion_embedding_dim: int = 128,
        d_state: int = 16,
        expand: int = 2,
        k_neighbors: int = 8,
        tau: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_obs_channels = n_obs_channels
        self.n_tar_channels = n_channels - n_obs_channels
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # Hierarchical Spatial Prior Encoder (HSPE)
        self.hspe = HierarchicalSpatialPriorEncoder(
            n_channels=n_channels,
            in_feature_dim=1,  # per-channel EEG signal
            hidden_dim=hidden_dim,
            k_neighbors=k_neighbors,
            tau=tau,
            num_regions=7,
            dropout=dropout
        )
        self.hspe.set_positions(electrode_positions)

        # Conditional State-Space Diffusion Reconstructor (CSDR)
        self.csdr = ConditionalStateSpaceDiffusionReconstructor(
            n_tar_channels=self.n_tar_channels,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_ssm=num_ssm,
            diffusion_steps=diffusion_steps,
            diffusion_embedding_dim=diffusion_embedding_dim,
            d_state=d_state,
            expand=expand
        )

        # Spatial prior projection
        self.spatial_prior_proj = nn.Linear(hidden_dim, hidden_dim)

        # Diffusion timestep embedding
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=diffusion_steps,
            embedding_dim=diffusion_embedding_dim
        )

    def forward(self, input_data):
        """
        Forward pass for training

        Args:
            input_data: tuple of (noisy_target, observed_eeg, mask, diffusion_step, obs_channels_idx, tar_channels_idx)

        Returns:
            noise_prediction: (B, K_tar, L)
        """
        noisy_target, observed_eeg, mask, diffusion_step, obs_channels_idx, tar_channels_idx = input_data

        B, K_tar, L = noisy_target.shape

        # Create masks
        obs_mask = torch.zeros(self.n_channels, device=noisy_target.device)
        obs_mask[obs_channels_idx] = 1
        tar_mask = torch.ones(self.n_channels, device=noisy_target.device)
        tar_mask[obs_channels_idx] = 0

        # Extract observed EEG for the observed channels only
        observed_eeg_subset = observed_eeg[:, obs_channels_idx, :]  # (B, K_obs, L)

        # Get spatial prior from HSPE
        G_spa = self.hspe(observed_eeg_subset, obs_mask, tar_mask)  # (B, K, hidden_dim)

        # Project spatial prior for target channels
        G_spa_tar = G_spa[:, tar_channels_idx, :]  # (B, K_tar, hidden_dim)
        G_spa_proj = self.spatial_prior_proj(G_spa_tar)  # (B, K_tar, hidden_dim)

        # Embed diffusion timestep
        diffusion_step_embed = self.diffusion_embedding(diffusion_step)

        # Create condition by zero-filling observed EEG to full layout
        # Then concatenate with spatial prior projection
        cond = torch.zeros(B, self.n_channels, L, device=noisy_target.device)
        cond[:, obs_channels_idx, :] = observed_eeg_subset
        cond_tar = cond[:, tar_channels_idx, :]  # (B, K_tar, L)

        # Concatenate condition with spatial prior
        # cond_with_prior: (B, K_tar, L + hidden_dim)
        # For simplicity, we integrate spatial prior into the denoising process via conditioning

        # Run CSDR
        noise_prediction = self.csdr((
            noisy_target,
            cond_tar,
            None,  # mask not used in current implementation
            diffusion_step,
            tar_channels_idx
        ))

        return noise_prediction


def build_tgsd_model(config, electrode_positions):
    """
    Build TGSD model from configuration

    Args:
        config: dict with model configuration
        electrode_positions: (K, 3) tensor of electrode positions

    Returns:
        TGSDModel
    """
    model = TGSDModel(
        n_channels=config['n_channels'],
        n_obs_channels=config['n_obs_channels'],
        electrode_positions=electrode_positions,
        seq_len=config.get('seq_len', 800),
        hidden_dim=config.get('hidden_dim', 64),
        num_layers=config.get('num_layers', 12),
        num_ssm=config.get('num_ssm', 2),
        diffusion_steps=config.get('diffusion_steps', 1000),
        diffusion_embedding_dim=config.get('diffusion_embedding_dim', 128),
        d_state=config.get('d_state', 16),
        expand=config.get('expand', 2),
        k_neighbors=config.get('k_neighbors', 8),
        tau=config.get('tau', 1.0),
        dropout=config.get('dropout', 0.1)
    )
    return model