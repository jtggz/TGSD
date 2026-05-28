"""
Utility functions for TGSD
"""
import os
import numpy as np
import torch
import logging


def flatten(v):
    """Flatten a list of lists/tuples"""
    return [x for y in v for x in y]


def find_max_epoch(path):
    """Find maximum epoch/iteration in path, formatted ${n_iter}.pkl"""
    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:] == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch


def print_size(net):
    """Print the number of parameters of a network"""
    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        logging.info("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6))


def std_normal(size, device='cuda'):
    """Generate the standard Gaussian variable of a certain size"""
    return torch.normal(0, 1, size=size).to(device)


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int): number of diffusion steps
    beta_0 and beta_T (float): beta schedule start/end value

    Returns:
    a dictionary of diffusion hyperparameters
    """
    Beta = torch.linspace(beta_0, beta_T, T)
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])
    Sigma = torch.sqrt(Beta_tilde)

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    return _dh


def sampling(net, size, diffusion_hyperparams, cond, mask, obs_channels_idx, tar_channels_idx, only_generate_missing=1):
    """
    Perform the complete sampling step for EEG spatial super-resolution

    Parameters:
    net: the TGSD model
    size: size of tensor to be generated (batch, channels, length)
    diffusion_hyperparams: dictionary of diffusion hyperparameters
    cond: condition (observed EEG)
    mask: mask indicating observed channels (1) and target channels (0)
    obs_channels_idx: indices of observed channels
    tar_channels_idx: indices of target channels
    only_generate_missing: if 1, only generate missing channels

    Returns:
    the generated EEG in torch.tensor
    """
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]

    x = std_normal(size, device=cond.device)

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            if only_generate_missing == 1:
                # Keep observed channels unchanged
                x_obs = cond[:, obs_channels_idx, :]
                x = torch.cat([x_obs, x], dim=1)

            diffusion_steps = (t * torch.ones((size[0], 1)).to(cond.device))
            epsilon_theta = net((x, cond, mask, diffusion_steps, tar_channels_idx))

            # Update x_{t-1} to mu_theta(x_t)
            x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
            if t > 0:
                x = x + Sigma[t] * std_normal(size, device=cond.device)

            if only_generate_missing == 1:
                # Restore observed channels
                x_obs_new = x[:, obs_channels_idx, :]
                x_tar = x[:, tar_channels_idx, :]
                x = torch.cat([x_obs_new, x_tar], dim=1)

    return x


def training_loss(net, loss_fn, X, diffusion_hyperparams, tar_channels_idx, only_generate_missing=1):
    """
    Compute the training loss

    Parameters:
    net: the TGSD model
    loss_fn: the loss function
    X: tuple of (audio/target, condition, mask, loss_mask)
    diffusion_hyperparams: dictionary of diffusion hyperparameters
    tar_channels_idx: indices of target channels
    only_generate_missing: if 1, only compute loss for missing channels

    Returns:
    training loss
    """
    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    target = X[0]
    cond = X[1]
    mask = X[2]
    loss_mask = X[3]

    B, C, L = target.shape
    diffusion_steps = torch.randint(T, size=(B, 1)).to(target.device)

    z = std_normal(target.shape, device=target.device)
    if only_generate_missing == 1:
        z = target * mask.float() + z * (1 - mask).float()

    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * target + torch.sqrt(
        1 - Alpha_bar[diffusion_steps]) * z

    # Extract target channels for prediction
    transformed_X_tar = transformed_X[:, tar_channels_idx, :]
    z_tar = z[:, tar_channels_idx, :]

    epsilon_theta = net((transformed_X_tar, cond, mask, diffusion_steps.view(B, 1), tar_channels_idx))

    if only_generate_missing == 1:
        return loss_fn(epsilon_theta[loss_mask[:, tar_channels_idx, :]], z_tar[loss_mask[:, tar_channels_idx, :]])
    elif only_generate_missing == 0:
        return loss_fn(epsilon_theta, z_tar)


def get_mask_channel_level(K_obs, K_tar, K_total, device='cuda'):
    """
    Create channel-level mask for EEG spatial super-resolution

    Args:
        K_obs: number of observed channels
        K_tar: number of target channels
        K_total: total number of channels
        device: device

    Returns:
        mask: (K_total,) tensor where 1=observed, 0=target
    """
    mask = torch.zeros(K_total, device=device)
    # First K_obs channels are observed
    mask[:K_obs] = 1
    return mask


def get_full_layout_mask(obs_channels_idx, K_total, device='cuda'):
    """
    Create mask for full electrode layout

    Args:
        obs_channels_idx: indices of observed channels
        K_total: total number of channels
        device: device

    Returns:
        mask: (K_total,) tensor where 1=observed, 0=target
    """
    mask = torch.zeros(K_total, device=device)
    mask[obs_channels_idx] = 1
    return mask