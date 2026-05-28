"""
Inference script for TGSD
"""
import os
import argparse
import json
import numpy as np
import torch
import logging
import datetime

from model.TGSD import TGSDModel
from dataset.eeg_dataset import EEGSpatialSuperResDataset, get_standard_electrode_positions
from utils.util import find_max_epoch, print_size, calc_diffusion_hyperparams, std_normal


def generate(
    output_directory,
    gen_config,
    model_config,
    trainset_config,
    diffusion_config,
    ckpt_iter='max',
    num_samples=100,
    batch_size=8
):
    """
    Generate/Evaluate TGSD model
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
        filename=os.path.join(output_directory, 'inference.log'),
        filemode='w'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Output directory: {output_directory}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info("Loading dataset...")
    test_dataset = EEGSpatialSuperResDataset(
        data_path=trainset_config['test_data_path'],
        super_resolution_factor=model_config.get('super_resolution_factor', 2),
        channel_layout_case=model_config.get('channel_layout_case', 1),
        seq_len=model_config.get('seq_len', 800),
        normalize=True
    )

    n_channels = test_dataset.n_channels
    n_obs_channels = test_dataset.n_obs_channels
    n_tar_channels = test_dataset.n_tar_channels
    seq_len = model_config.get('seq_len', 800)

    logger.info(f"Test dataset: {len(test_dataset)} samples, {n_channels} channels")

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
        dropout=0.0  # No dropout during inference
    )
    model = model.to(device)
    print_size(model)

    # Load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
        if ckpt_iter < 0:
            logger.warning("No checkpoint found!")
            return

    model_path = os.path.join(output_directory, f'{ckpt_iter}.pkl')
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info(f'Successfully loaded model at iteration {ckpt_iter}')
    except Exception as e:
        logger.warning(f'Failed to load model: {e}')
        return

    # Diffusion hyperparameters
    diffusion_hyperparams = calc_diffusion_hyperparams(
        diffusion_config['T'],
        diffusion_config['beta_0'],
        diffusion_config['beta_T']
    )
    for key in diffusion_hyperparams:
        if key != 'T':
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)

    # Evaluation
    model.eval()
    logger.info("Starting evaluation...")

    all_nmse = []
    all_pcc = []

    with torch.no_grad():
        for sample_idx in range(min(num_samples, len(test_dataset))):
            full_eeg, obs_eeg, tar_eeg, obs_mask, tar_mask, obs_idx, tar_idx = test_dataset[sample_idx]

            full_eeg = full_eeg.unsqueeze(0).to(device)
            obs_eeg = obs_eeg.unsqueeze(0).to(device)
            tar_eeg = tar_eeg.unsqueeze(0).to(device)
            obs_mask = obs_mask.unsqueeze(0).to(device)
            tar_mask = tar_mask.unsqueeze(0).to(device)
            obs_idx = obs_idx.to(device)
            tar_idx = tar_idx.to(device)

            B, K_obs, L = obs_eeg.shape
            _, K_tar, _ = tar_eeg.shape

            # Start from Gaussian noise
            x_t = std_normal((B, K_tar, L), device=device)

            # Diffusion sampling (reverse process)
            T = diffusion_config['T']
            Alpha = diffusion_hyperparams['Alpha']
            Alpha_bar = diffusion_hyperparams['Alpha_bar']
            Sigma = diffusion_hyperparams['Sigma']

            for t in range(T - 1, -1, -1):
                # Condition on observed channels
                diffusion_steps = (t * torch.ones((B, 1))).to(device)

                # Predict noise
                noise_pred = model((
                    x_t,
                    torch.cat([obs_eeg, tar_eeg], dim=1),
                    tar_mask,
                    diffusion_steps,
                    obs_idx,
                    tar_idx
                ))

                # Update x_{t-1}
                alpha_t = Alpha[t]
                alpha_bar_t = Alpha_bar[t]
                sigma_t = Sigma[t]

                x_t = (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_t)

                if t > 0:
                    x_t = x_t + sigma_t * std_normal((B, K_tar, L), device=device)

                # Keep observed channels unchanged (not implemented in this simplified version)

            # Compute metrics
            tar_eeg_np = tar_eeg.cpu().numpy()
            x_t_np = x_t.cpu().numpy()

            # NMSE
            numerator = np.sum((tar_eeg_np - x_t_np) ** 2)
            denominator = np.sum(tar_eeg_np ** 2)
            nmse = numerator / (denominator + 1e-8)

            # PCC
            pcc_sum = 0
            for ch in range(K_tar):
                corr = np.corrcoef(tar_eeg_np[0, ch], x_t_np[0, ch])[0, 1]
                pcc_sum += corr
            pcc = pcc_sum / K_tar

            all_nmse.append(nmse)
            all_pcc.append(pcc)

            if (sample_idx + 1) % 10 == 0:
                logger.info(f"Processed {sample_idx + 1}/{num_samples} samples")

    # Report results
    logger.info(f"Average NMSE: {np.mean(all_nmse):.4f} +/- {np.std(all_nmse):.4f}")
    logger.info(f"Average PCC: {np.mean(all_pcc):.4f} +/- {np.std(all_pcc):.4f}")

    # Save results
    results = {
        'nmse': all_nmse,
        'pcc': all_pcc,
        'mean_nmse': float(np.mean(all_nmse)),
        'mean_pcc': float(np.mean(all_pcc)),
        'std_nmse': float(np.std(all_nmse)),
        'std_pcc': float(np.std(all_pcc))
    }

    results_path = os.path.join(output_directory, f'results_iter{ckpt_iter}.json')
    import json as json_module
    with open(results_path, 'w') as f:
        json_module.dump(results, f)

    logger.info(f"Results saved to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_tgsd.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', type=str, default='max',
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=100,
                        help='Number of samples to evaluate')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    logging.info(config)

    gen_config = config['gen_config']
    model_config = config['model_config']
    trainset_config = config['trainset_config']
    diffusion_config = config['diffusion_config']

    generate(
        output_directory=gen_config.get('output_directory', 'results'),
        gen_config=gen_config,
        model_config=model_config,
        trainset_config=trainset_config,
        diffusion_config=diffusion_config,
        ckpt_iter=args.ckpt_iter,
        num_samples=args.num_samples,
        batch_size=gen_config.get('batch_size', 8)
    )