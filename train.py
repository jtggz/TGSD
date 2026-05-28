"""
Training script for TGSD
"""
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import logging
import datetime

from torch.utils.data import DataLoader
from model.TGSD import TGSDModel
from model.HSPE import HierarchicalSpatialPriorEncoder
from model.CSDR import ConditionalStateSpaceDiffusionReconstructor
from dataset.eeg_dataset import EEGSpatialSuperResDataset, get_standard_electrode_positions
from utils.util import (
    find_max_epoch, print_size, calc_diffusion_hyperparams,
    training_loss, std_normal
)


def train(
    output_directory,
    trainset_config,
    diffusion_config,
    model_config,
    n_iters=100000,
    iters_per_ckpt=10000,
    iters_per_logging=100,
    learning_rate=1e-4,
    batch_size=8,
    ckpt_iter=-1
):
    """
    Train TGSD model
    """
    # Create output directory
    local_path = "T{}_beta0{}_betaT{}_sr{}".format(
        diffusion_config["T"],
        diffusion_config["beta_0"],
        diffusion_config["beta_T"],
        model_config.get('super_resolution_factor', 2)
    )
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(output_directory, 'train.log'),
        filemode='w'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Output directory: {output_directory}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = EEGSpatialSuperResDataset(
        data_path=trainset_config['train_data_path'],
        super_resolution_factor=model_config.get('super_resolution_factor', 2),
        channel_layout_case=model_config.get('channel_layout_case', 1),
        seq_len=model_config.get('seq_len', 800),
        normalize=True
    )
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    n_channels = dataset.n_channels
    n_obs_channels = dataset.n_obs_channels
    n_tar_channels = dataset.n_tar_channels
    seq_len = model_config.get('seq_len', 800)

    logger.info(f"Dataset: {len(dataset)} samples, {n_channels} channels, {n_obs_channels} observed, {n_tar_channels} target")

    # Get electrode positions
    electrode_positions = torch.from_numpy(get_standard_electrode_positions(n_channels)).float()
    electrode_positions = electrode_positions.to(device)

    # Build model
    logger.info("Building TGSD model...")
    model = TGSDModel(
        n_channels=n_channels,
        n_obs_channels=n_obs_channels,
        electrode_positions=electrode_positions,
        seq_len=seq_len,
        hidden_dim=model_config.get('hidden_dim', 64),
        num_layers=model_config.get('num_layers', 12),
        num_ssm=model_config.get('num_ssm', 2),
        diffusion_steps=diffusion_config['T'],
        diffusion_embedding_dim=model_config.get('diffusion_embedding_dim', 128),
        d_state=model_config.get('d_state', 16),
        expand=model_config.get('expand', 2),
        k_neighbors=model_config.get('k_neighbors', 8),
        tau=model_config.get('tau', 1.0),
        dropout=model_config.get('dropout', 0.1)
    )
    model = model.to(device)
    print_size(model)

    # Load checkpoint if exists
    if ckpt_iter >= 0:
        try:
            model_path = os.path.join(output_directory, f'{ckpt_iter}.pkl')
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f'Successfully loaded model at iteration {ckpt_iter}')
        except:
            ckpt_iter = -1
            logger.info('No valid checkpoint found, starting from initialization')
    else:
        ckpt_iter = -1
        logger.info('No valid checkpoint found, starting from initialization')

    # Diffusion hyperparameters
    diffusion_hyperparams = calc_diffusion_hyperparams(
        diffusion_config['T'],
        diffusion_config['beta_0'],
        diffusion_config['beta_T']
    )
    # Move to device
    for key in diffusion_hyperparams:
        if key != 'T':
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    logger.info("Starting training...")
    n_iter = ckpt_iter + 1
    model.train()

    while n_iter <= n_iters:
        for batch_data in train_loader:
            full_eeg, obs_eeg, tar_eeg, obs_mask, tar_mask, obs_idx, tar_idx = batch_data

            full_eeg = full_eeg.to(device)
            obs_eeg = obs_eeg.to(device)
            tar_eeg = tar_eeg.to(device)
            obs_mask = obs_mask.to(device)
            tar_mask = tar_mask.to(device)
            obs_idx = obs_idx.to(device)
            tar_idx = tar_idx.to(device)

            B, K_obs, L = obs_eeg.shape
            _, K_tar, _ = tar_eeg.shape

            # Prepare input for model
            # Noisy target for training
            diffusion_steps = torch.randint(
                diffusion_config['T'],
                size=(B, 1)
            ).to(device)

            # Sample noise
            z = std_normal((B, K_tar, L), device=device)

            # Forward diffusion to get noisy target
            alpha_bar = diffusion_hyperparams['Alpha_bar'][diffusion_steps]  # (B, 1, 1)
            noisy_tar = torch.sqrt(alpha_bar) * tar_eeg + torch.sqrt(1 - alpha_bar) * z

            # Forward pass
            noise_pred = model((
                noisy_tar,
                torch.cat([obs_eeg, tar_eeg], dim=1),  # Full EEG as condition (simplified)
                tar_mask,
                diffusion_steps,
                obs_idx,
                tar_idx
            ))

            # Compute loss
            loss = nn.functional.mse_loss(noise_pred, z)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                logger.info(f"Iteration {n_iter}: Loss = {loss.item():.6f}")

            if n_iter % iters_per_ckpt == 0:
                checkpoint_path = os.path.join(output_directory, f'{n_iter}.pkl')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iteration': n_iter,
                    'loss': loss.item()
                }, checkpoint_path)
                logger.info(f'Model saved at iteration {n_iter}')

            n_iter += 1
            if n_iter > n_iters:
                break

    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_tgsd.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', type=int, default=-1,
                        help='Checkpoint iteration to resume from')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    logging.info(config)

    train_config = config['train_config']
    trainset_config = config['trainset_config']
    diffusion_config = config['diffusion_config']
    model_config = config['model_config']

    train(
        output_directory=train_config.get('output_directory', 'checkpoints'),
        trainset_config=trainset_config,
        diffusion_config=diffusion_config,
        model_config=model_config,
        n_iters=train_config.get('n_iters', 100000),
        iters_per_ckpt=train_config.get('iters_per_ckpt', 10000),
        iters_per_logging=train_config.get('iters_per_logging', 100),
        learning_rate=train_config.get('learning_rate', 1e-4),
        batch_size=train_config.get('batch_size', 8),
        ckpt_iter=args.ckpt_iter
    )