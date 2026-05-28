"""
EEG Dataset for TGSD spatial super-resolution
"""
import torch
import numpy as np
from torch.utils.data import Dataset
import random


class EEGDataset(Dataset):
    """
    Base EEG Dataset for loading preprocessed EEG data
    """
    def __init__(self, data_path, seq_len=None, normalize=True):
        """
        Args:
            data_path: path to numpy array of EEG data (N, K, L) or (N, L, K)
            seq_len: sequence length for training (if None, use full length)
            normalize: whether to normalize the data
        """
        self.data = np.load(data_path)
        # Ensure data is (N, K, L) format
        if self.data.shape[1] < self.data.shape[2]:
            self.data = np.transpose(self.data, (0, 2, 1))
        self.seq_len = seq_len if seq_len else self.data.shape[2]
        self.normalize = normalize
        self.mean = None
        self.std = None

        if normalize:
            self.mean = self.data.mean()
            self.std = self.data.std()
            self.data = (self.data - self.mean) / (self.std + 1e-8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        eeg = self.data[index]  # (K, L)
        K, L = eeg.shape

        # Random segment selection
        if L > self.seq_len:
            start = random.randint(0, L - self.seq_len)
            eeg = eeg[:, start:start + self.seq_len]

        return torch.from_numpy(eeg).float()


class EEGSpatialSuperResDataset(Dataset):
    """
    EEG Dataset for Spatial Super-Resolution

    Simulates low-density EEG acquisition by selecting subsets of channels
    as observed channels and treating the rest as target channels to reconstruct.

    Args:
        data_path: path to numpy array of EEG data (N, K, L)
        super_resolution_factor: spatial super-resolution factor (2, 4, or 8)
        channel_layout_case: predefined layout case (1-4) for observed/target partition
        seq_len: sequence length for training
        normalize: whether to normalize the data
    """
    def __init__(
        self,
        data_path,
        super_resolution_factor=2,
        channel_layout_case=1,
        seq_len=None,
        normalize=True
    ):
        self.data = np.load(data_path)
        # Ensure data is (N, K, L) format
        if self.data.shape[1] < self.data.shape[2]:
            self.data = np.transpose(self.data, (0, 2, 1))

        self.n_samples, self.n_channels, self.seq_len_full = self.data.shape
        self.seq_len = seq_len if seq_len else self.seq_len_full
        self.normalize = normalize
        self.super_resolution_factor = super_resolution_factor

        # Calculate observed and target channels
        self.n_obs_channels = self.n_channels // super_resolution_factor
        self.n_tar_channels = self.n_channels - self.n_obs_channels

        # Get channel partition based on layout case
        self.obs_channels_idx, self.tar_channels_idx = self.get_channel_partition(
            channel_layout_case
        )

        if normalize:
            self.mean = self.data.mean()
            self.std = self.data.std()
            self.data = (self.data - self.mean) / (self.std + 1e-8)

    def get_channel_partition(self, case):
        """
        Get observed and target channel indices based on predefined layouts.

        Following Deep-EEGSR/Tang et al. 2022 protocol:
        - Case 1-4: Different predefined low-density electrode layouts
        - 2x: 31 observed + 31 target channels
        - 4x: 15 observed + 47 target channels
        - 8x: 8 observed + 54 target channels

        Args:
            case: layout case number (1-4)

        Returns:
            obs_channels_idx: indices of observed channels
            tar_channels_idx: indices of target channels
        """
        # Check if we can use predefined case or need to generate
        if not hasattr(self, 'super_resolution_factor'):
            # Default fallback
            obs_indices = list(range(0, self.n_channels, 2))
            tar_indices = list(set(range(self.n_channels)) - set(obs_indices))
            return obs_indices, tar_indices

        sr = self.super_resolution_factor

        # For 62 channels
        if self.n_channels == 62:
            if sr == 2:
                # 31 observed, 31 target - use alternating pattern
                partitions = {
                    1: {
                        'obs': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60],
                        'tar': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61]
                    },
                    2: {
                        # Another layout pattern - evenly distributed
                        'obs': [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29],
                        'tar': [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]
                    },
                    3: {
                        # Third pattern
                        'obs': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58],
                        'tar': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59]
                    },
                    4: {
                        # Fourth pattern
                        'obs': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 3, 8, 13, 18, 23],
                        'tar': [2, 4, 7, 9, 12, 14, 17, 19, 22, 24, 27, 29, 32, 34, 37, 39, 42, 44, 47, 49, 52, 54, 57, 59, 3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58]
                    }
                }
            elif sr == 4:
                # 15 observed, 47 target
                partitions = {
                    1: {
                        'obs': [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56],
                        'tar': [i for i in range(62) if i not in [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56]]
                    },
                    2: {
                        'obs': [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57],
                        'tar': [i for i in range(62) if i not in [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57]]
                    },
                    3: {
                        'obs': [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58],
                        'tar': [i for i in range(62) if i not in [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58]]
                    },
                    4: {
                        'obs': [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59],
                        'tar': [i for i in range(62) if i not in [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59]]
                    }
                }
            elif sr == 8:
                # 8 observed, 54 target
                partitions = {
                    1: {
                        'obs': [0, 8, 16, 24, 32, 40, 48, 56],
                        'tar': [i for i in range(62) if i not in [0, 8, 16, 24, 32, 40, 48, 56]]
                    },
                    2: {
                        'obs': [1, 9, 17, 25, 33, 41, 49, 57],
                        'tar': [i for i in range(62) if i not in [1, 9, 17, 25, 33, 41, 49, 57]]
                    },
                    3: {
                        'obs': [2, 10, 18, 26, 34, 42, 50, 58],
                        'tar': [i for i in range(62) if i not in [2, 10, 18, 26, 34, 42, 50, 58]]
                    },
                    4: {
                        'obs': [3, 11, 19, 27, 35, 43, 51, 59],
                        'tar': [i for i in range(62) if i not in [3, 11, 19, 27, 35, 43, 51, 59]]
                    }
                }
            else:
                raise ValueError(f"Super-resolution factor {sr} not supported")
        else:
            # Default fallback for other channel counts
            obs_indices = list(range(0, self.n_channels, sr))
            tar_indices = list(set(range(self.n_channels)) - set(obs_indices))
            partitions = {1: {'obs': obs_indices, 'tar': tar_indices}}

        if case in partitions:
            return partitions[case]['obs'], partitions[case]['tar']
        else:
            return partitions[1]['obs'], partitions[1]['tar']

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        """
        Returns:
            full_eeg: (K, L) - full channel EEG
            obs_eeg: (K_obs, L) - observed channels only
            tar_eeg: (K_tar, L) - target channels (for loss computation)
            obs_mask: (K,) - mask indicating observed channels (1) and target channels (0)
            tar_mask: (K,) - mask indicating target channels (1) and observed channels (0)
        """
        eeg = self.data[index]  # (K, L_full)

        # Segment selection
        if self.seq_len_full > self.seq_len:
            start = random.randint(0, self.seq_len_full - self.seq_len)
            eeg = eeg[:, start:start + self.seq_len]

        # Extract observed and target channels
        obs_eeg = eeg[self.obs_channels_idx, :]
        tar_eeg = eeg[self.tar_channels_idx, :]

        # Create masks
        obs_mask = torch.zeros(self.n_channels, dtype=torch.float32)
        obs_mask[self.obs_channels_idx] = 1.0

        tar_mask = torch.zeros(self.n_channels, dtype=torch.float32)
        tar_mask[self.tar_channels_idx] = 1.0

        return (
            torch.from_numpy(eeg).float(),
            torch.from_numpy(obs_eeg).float(),
            torch.from_numpy(tar_eeg).float(),
            obs_mask,
            tar_mask,
            torch.tensor(self.obs_channels_idx),
            torch.tensor(self.tar_channels_idx)
        )


def get_standard_electrode_positions(n_channels=62):
    """
    Get standard 3D electrode positions for EEG caps (10-20 system)

    Args:
        n_channels: number of electrodes (62 for standard 10-20 system)

    Returns:
        positions: (n_channels, 3) numpy array of 3D coordinates
    """
    if n_channels == 62:
        # Standard 10-20 system 62-channel positions (from PGCN)
        positions = np.array([
            (-2.285379, 10.372299, 4.564709),  # 0: Fp1
            (0.687462, 10.931931, 4.452579),   # 1: FpZ
            (3.874373, 9.896583, 4.368097),   # 2: Fp2
            (-2.82271, 9.895013, 6.833403),   # 3: AF7
            (4.143959, 9.607678, 7.067061),   # 4: AF8
            (-6.417786, 6.362997, 4.476012),  # 5: F3
            (-5.745505, 7.282387, 6.764246),  # 6: F4
            (-4.248579, 7.990933, 8.73188),   # 7: Fz
            (-2.046628, 8.049909, 10.162745), # 8: F1
            (0.716282, 7.836015, 10.88362),   # 9: F2
            (3.193455, 7.889754, 10.312743),  # 10: F5
            (5.337832, 7.691511, 8.678795),   # 11: F6
            (6.842302, 6.643506, 6.300108),   # 12: F7
            (7.197982, 5.671902, 4.245699),   # 13: F8
            (-7.326021, 3.749974, 4.734323),  # 14: FC3
            (-6.882368, 4.211114, 7.939393),  # 15: FC4
            (-4.837038, 4.672796, 10.955297), # 16: FCz
            (-2.677567, 4.478631, 12.365311),  # 17: FC1
            (0.455027, 4.186858, 13.104445),   # 18: FC2
            (3.654295, 4.254963, 12.205945),  # 19: FC5
            (5.863695, 4.275586, 10.714709),  # 20: FC6
            (7.610693, 3.851083, 7.604854),   # 21: FT7
            (7.821661, 3.18878, 4.400032),     # 22: FT8
            (-7.640498, 0.756314, 4.967095),  # 23: T7
            (-7.230136, 0.725585, 8.331517),  # 24: TP7
            (-5.748005, 0.480691, 11.193904), # 25: TP9
            (-3.009834, 0.621885, 13.441012), # 26: C5
            (0.341982, 0.449246, 13.839247),  # 27: C3
            (3.62126, 0.31676, 13.082255),    # 28: C4
            (6.418348, 0.200262, 11.178412),  # 29: C6
            (7.743287, 0.254288, 8.143276),   # 30: TP10
            (8.214926, 0.533799, 4.980188),   # 31: T8
            (-7.794727, -1.924366, 4.686678), # 32: TTP7
            (-7.103159, -2.735806, 7.908936), # 33: TP8
            (-5.549734, -3.131109, 10.995642), # 34: C1
            (-3.111164, -3.281632, 12.904391), # 35: Cz
            (-0.072857, -3.405421, 13.509398), # 36: C2
            (3.044321, -3.820854, 12.781214),  # 37: CCP5
            (5.712892, -3.643826, 10.907982),  # 38: CCP6
            (7.304755, -3.111501, 7.913397),  # 39: CP5
            (7.92715, -2.443219, 4.673271),   # 40: CP6
            (-7.161848, -4.799244, 4.411572), # 41: P5
            (-6.375708, -5.683398, 7.142764),  # 42: P6
            (-5.117089, -6.324777, 9.046002), # 43: P3
            (-2.8246, -6.605847, 10.717917),  # 44: P4
            (-0.19569, -6.696784, 11.505725), # 45: Pz
            (2.396374, -7.077637, 10.585553), # 46: P1
            (4.802065, -6.824497, 8.991351),  # 47: P2
            (6.172683, -6.209247, 7.028114),  # 48: P7
            (7.187716, -4.954237, 4.477674),  # 49: P8
            (-5.894369, -6.974203, 4.318362), # 50: PO7
            (-5.037746, -7.566237, 6.585544),  # 51: PO8
            (-2.544662, -8.415612, 7.820205),  # 52: PO3
            (-0.339835, -8.716856, 8.249729),  # 53: PO4
            (2.201964, -8.66148, 7.796194),    # 54: POz
            (4.491326, -8.16103, 6.387415),   # 55: O1
            (5.766648, -7.498684, 4.546538),   # 56: O2
            (-6.387065, -5.755497, 1.886141),  # 57: PO9
            (-3.542601, -8.904578, 4.214279),  # 58: PO10
            (-0.080624, -9.660508, 4.670766),  # 59: Oz
            (3.050584, -9.25965, 4.194428),    # 60: Iz
            (6.192229, -6.797348, 2.355135),   # 61: I2
        ], dtype=np.float32)
        return positions
    else:
        raise ValueError(f"Only 62 channels is supported, got {n_channels}")