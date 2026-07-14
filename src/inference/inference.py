import sys
import os

import json
import scipy.io
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.training.args import parse_args
from src.model.mamba import PolarizationMamba, PolarizationMambaSO3
from src.model.loss import AngularLoss, PoincareRegularizedMSE, Infidelity
from src.utils.plotting import output_results
import platform
import random

# Check system OS for appropriate Mamba implementation
system_os = platform.system()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch version:", torch.__version__)
print("Current device:", device)

class SParameterDataset(Dataset):
    def __init__(self, features, targets, window_size):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.window_size = window_size

    def __len__(self):
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]
        y = self.targets[idx + self.window_size].unsqueeze(0)
        return x, y

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    # Optional: forces deterministic algorithms (can slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

if __name__ == '__main__':

    #set random seed 
    set_seed(42)

    args = parse_args()
    window_size = args.window_size
    batch_size = args.batch_size
    lr = args.lr
    delta_lambda = args.wavelength_range
    lambda_reg = args.lambda_reg

    loss_constructors = {
            "mse": lambda: nn.MSELoss(),
            "regmse": lambda: PoincareRegularizedMSE(lambda_reg=lambda_reg if lambda_reg is not None else 0.2),
            "angular": lambda: AngularLoss(lambda_reg=lambda_reg if lambda_reg is not None else 0.02),
            "infidelity": lambda: Infidelity(),
        }

    loss_type = loss_constructors[args.loss.lower()]()

    dataset_paths = {
            "synthetic_1mm": "data/synthetic/100k_samples_txp_1551.5_pax_1552.5_polcon_and_fiber_1Hz.mat",
            "synthetic_5mm": "data/synthetic/400k_samples_txp_1551.5_pax_1556.5_polcon_and_fiber_2_1Hz.mat",
            "synthetic_10mm": "data/synthetic/400k_samples_txp_1551.5_pax_1561.5_polcon_and_fiber_2_1Hz.mat",
            "synthetic_14mm": "data/synthetic/400k_samples_txp_1551.5_pax_1565.5_polcon_and_fiber_2_1Hz.mat",
            "synthetic_-5mm": "data/synthetic/400k_samples_txp_1551.5_pax_1546.5_polcon_and_fiber_2_1Hz.mat",
            "loop_1mm": "data/chicago_loop/txp_1551.5_pax_1552.5_fiber_loop.mat",
            "loop_5mm": "data/chicago_loop/txp_1551.5_pax_1556.5_fiber_loop.mat",
            "loop_10mm": "data/chicago_loop/txp_1551.5_pax_1561.5_fiber_loop.mat",
            "loop_14mm": "data/chicago_loop/txp_1551.5_pax_1565.5_fiber_loop.mat",
        }

    path = dataset_paths.get(delta_lambda)
    if path is None:
        print(f"Dataset does not exist for wavelength range {delta_lambda}")
        exit(1)
        
    if not os.path.exists(path):
        print(f"File not found at {path}")
        exit(1)
    else:
        print(f"Loading data from {path}")
        mat_data = scipy.io.loadmat(path)

    # Extract Data
    s1_pax = mat_data['s1_pax'].flatten()
    s2_pax = mat_data['s2_pax'].flatten()
    s3_pax = mat_data['s3_pax'].flatten()
    s1_txp = mat_data['s1_txp'].flatten()
    s2_txp = mat_data['s2_txp'].flatten()
    s3_txp = mat_data['s3_txp'].flatten()

    input_data = np.column_stack([s1_txp, s2_txp, s3_txp])
    known_output = np.column_stack([s1_pax, s2_pax, s3_pax])