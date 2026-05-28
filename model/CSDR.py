"""
Conditional State-Space Diffusion Reconstructor (CSDR) for TGSD

Reconstructs missing-channel EEG through conditional reverse diffusion,
using alternating temporal and channel-wise state-space modeling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from .mamba import Mamba, MambaBlock, RMSNorm


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    """Conv1d with Kaiming initialization"""
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    layer = nn.utils.weight_norm(layer)
    nn.init.kaiming_normal_(layer.weight)
    return layer


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


def cal_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """Embed diffusion steps into higher dimensional space"""
    assert diffusion_step_embed_dim_in % 2 == 0
    half_dim = diffusion_step_embed_dim_in // 2
    _embed = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim, device=diffusion_steps.device) * -_embed)
    _embed = diffusion_steps.float() * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed), torch.cos(_embed)), 1)
    return diffusion_step_embed


class TemporalSSM(nn.Module):
    """
    Temporal State-Space Model block
    Captures long-range temporal dependencies within each channel
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1
        )

        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + d_model // 16, bias=False)
        self.dt_proj = nn.Linear(d_model // 16, self.d_inner, bias=True)

        # Initialize
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, K, L, D) - [batch, channels, seq_len, dim]

        Returns:
            output: (B, K, L, D)
        """
        B, K, L, D = x.shape

        # Merge channel dimension for temporal processing
        x = rearrange(x, 'b k l d -> (b k) l d')

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)  # (B*K, D_inner, L)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)  # (B*K, L, D_inner)

        x = F.silu(x)
        y = self.ssm(x)

        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)

        # Restore shape
        output = rearrange(output, '(b k) l d -> b k l d', b=B, k=K)
        return output

    def ssm(self, x):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)
        delta, B_state, C_state = torch.split(deltaBC, [self.d_model // 16, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        # Selective scan
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B_state.unsqueeze(2)

        # Simple scan (not parallel for simplicity)
        h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=x.device)
        hs = []
        for t in range(x.size(1)):
            h = deltaA[:, t] * h + deltaB[:, t]
            hs.append(h)
        hs = torch.stack(hs, dim=1)

        y = torch.einsum('b l d n, b l n -> b l d', hs, C_state)
        y = y + D * x

        return y


class ChannelSSM(nn.Module):
    """
    Channel-wise State-Space Model block
    Captures cross-channel dependencies by swapping temporal and channel dimensions
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1
        )

        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + d_model // 16, bias=False)
        self.dt_proj = nn.Linear(d_model // 16, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, K, L, D) - [batch, channels, seq_len, dim]

        Returns:
            output: (B, K, L, D)
        """
        B, K, L, D = x.shape

        # Swap channel and time dimensions: (B, K, L, D) -> (B, L, K, D)
        x = rearrange(x, 'b k l d -> b l k d')

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)  # (B, D_inner, K)
        x = self.conv1d(x)[:, :, :K]
        x = x.transpose(1, 2)  # (B, K, D_inner)

        x = F.silu(x)
        y = self.ssm(x)

        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)

        # Restore shape: (B, L, K, D) -> (B, K, L, D)
        output = rearrange(output, 'b l k d -> b k l d')

        return output

    def ssm(self, x):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)
        delta, B_state, C_state = torch.split(deltaBC, [self.d_model // 16, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B_state.unsqueeze(2)

        h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=x.device)
        hs = []
        for t in range(x.size(1)):
            h = deltaA[:, t] * h + deltaB[:, t]
            hs.append(h)
        hs = torch.stack(hs, dim=1)

        y = torch.einsum('b l d n, b l n -> b l d', hs, C_state)
        y = y + D * x

        return y


class DenoisingBlock(nn.Module):
    """
    Single denoising block with alternating temporal and channel SSM
    """
    def __init__(self, hidden_dim, num_temporal_ssm=1, num_channel_ssm=1, d_state=16, expand=2):
        super().__init__()

        self.temporal_ssm = nn.ModuleList([
            TemporalSSM(hidden_dim, d_state=d_state, expand=expand)
            for _ in range(num_temporal_ssm)
        ])

        self.channel_ssm = nn.ModuleList([
            ChannelSSM(hidden_dim, d_state=d_state, expand=expand)
            for _ in range(num_channel_ssm)
        ])

        self.norm_temp = RMSNorm(hidden_dim)
        self.norm_ch = RMSNorm(hidden_dim)

    def forward(self, z):
        """
        Args:
            z: (B, K_tar, L, D) - denoising state

        Returns:
            output: (B, K_tar, L, D)
        """
        # Temporal SSM
        for temp_ssm in self.temporal_ssm:
            z = z + temp_ssm(self.norm_temp(z))

        # Channel SSM
        for ch_ssm in self.channel_ssm:
            z = z + ch_ssm(self.norm_ch(z))

        return z


class ConditionalStateSpaceDiffusionReconstructor(nn.Module):
    """
    Conditional State-Space Diffusion Reconstructor (CSDR)

    Reconstructs missing-channel EEG through conditional reverse diffusion,
    guided by spatial priors and observed EEG signals.

    Args:
        n_tar_channels: Number of target channels to reconstruct
        seq_len: Sequence length
        hidden_dim: Hidden dimension for the model
        num_layers: Number of denoising blocks
        num_ssm: Number of SSM layers per block
        diffusion_steps: Number of diffusion steps
        diffusion_embedding_dim: Dimension for diffusion step embedding
        d_state: SSM state dimension
        expand: SSM expansion factor
    """
    def __init__(
        self,
        n_tar_channels: int,
        seq_len: int,
        hidden_dim: int = 64,
        num_layers: int = 12,
        num_ssm: int = 2,
        diffusion_steps: int = 1000,
        diffusion_embedding_dim: int = 128,
        d_state: int = 16,
        expand: int = 2
    ):
        super().__init__()
        self.n_tar_channels = n_tar_channels
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Diffusion timestep embedding
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=diffusion_steps,
            embedding_dim=diffusion_embedding_dim
        )

        # Input projections
        self.input_proj = Conv1d_with_init(n_tar_channels, hidden_dim, 1)
        self.cond_proj = Conv1d_with_init(n_tar_channels, hidden_dim, 1)
        self.spatial_prior_proj = nn.Linear(hidden_dim, hidden_dim)

        # Denoising blocks with alternating temporal and channel SSM
        self.denoising_blocks = nn.ModuleList([
            DenoisingBlock(
                hidden_dim=hidden_dim,
                num_temporal_ssm=num_ssm,
                num_channel_ssm=num_ssm,
                d_state=d_state,
                expand=expand
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.out_proj = Conv1d_with_init(hidden_dim, n_tar_channels, 1)

        # Skip connections
        self.skip_proj = nn.ModuleList([
            Conv1d_with_init(hidden_dim, hidden_dim, 1)
            for _ in range(num_layers)
        ])

    def forward(self, input_data):
        """
        Forward pass for training

        Args:
            input_data: tuple of (noisy_target, condition, mask, diffusion_step, tar_channels_idx)

        Returns:
            noise_prediction: predicted noise (B, K_tar, L)
        """
        noisy_target, condition, mask, diffusion_step, tar_channels_idx = input_data

        B, K_tar, L = noisy_target.shape

        # Embed diffusion timestep
        diffusion_step_embed = self.diffusion_embedding(diffusion_step)  # (B, diffusion_embedding_dim)
        diffusion_step_embed = diffusion_step_embed.unsqueeze(-1)  # (B, diffusion_embedding_dim, 1)

        # Project condition (observed EEG)
        cond_proj = self.cond_proj(condition)  # (B, K, hidden_dim)
        cond_proj = cond_proj.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)  # (B, K, L, hidden_dim)

        # Project noisy target
        h = self.input_proj(noisy_target)  # (B, K_tar, hidden_dim)
        h = h.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)  # (B, K_tar, L, hidden_dim)

        # Add diffusion timestep embedding
        diff_embed = self.diffusion_embedding.proj1(diffusion_step_embed)  # (B, hidden_dim)
        diff_embed = F.silu(diff_embed)
        diff_embed = self.diffusion_embedding.proj2(diff_embed)  # (B, hidden_dim)
        h = h + diff_embed.unsqueeze(2).unsqueeze(2)

        # Denoising blocks
        skip = 0
        for n, block in enumerate(self.denoising_blocks):
            h = block(h)
            skip_n = self.skip_proj[n](h.transpose(1, 2)).transpose(1, 2)
            skip = skip + skip_n

        skip = skip / math.sqrt(self.num_layers)

        # Output projection
        out = self.out_proj(skip.transpose(1, 2)).transpose(1, 2)  # (B, K_tar, L)

        return out


class TGSD(nn.Module):
    """
    Complete TGSD model combining HSPE and CSDR

    Args:
        n_channels: Total number of channels (K)
        n_obs_channels: Number of observed channels
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        num_layers: Number of denoising blocks
        num_ssm: Number of SSM layers per block
        diffusion_steps: Number of diffusion steps
        diffusion_embedding_dim: Dimension for diffusion step embedding
        d_state: SSM state dimension
        expand: SSM expansion factor
    """
    def __init__(
        self,
        n_channels: int,
        n_obs_channels: int,
        seq_len: int = 800,
        hidden_dim: int = 64,
        num_layers: int = 12,
        num_ssm: int = 2,
        diffusion_steps: int = 1000,
        diffusion_embedding_dim: int = 128,
        d_state: int = 16,
        expand: int = 2
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_obs_channels = n_obs_channels
        self.n_tar_channels = n_channels - n_obs_channels
        self.seq_len = seq_len

        # Conditional State-Space Diffusion Reconstructor
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

    def forward(self, input_data):
        """
        Forward pass

        Args:
            input_data: tuple of (noisy_target, condition, mask, diffusion_step, tar_channels_idx)

        Returns:
            noise_prediction: (B, K_tar, L)
        """
        return self.csdr(input_data)