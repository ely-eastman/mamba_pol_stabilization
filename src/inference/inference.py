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
#system_os = platform.system()



class PolarizationPredictor():
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print("PyTorch version:", torch.__version__)
        print("Current device:", self.device)
        if self.device.type == 'cuda':
            system_os = 'Linux'
        if self.device.type == 'cpu':
            system_os = "CPU"
        saved_model = torch.load(model_path, map_location=self.device, weights_only=False)

        model_config = saved_model["config"]
        state_dict = saved_model["state_dict"]
        self.window_size = model_config["window_size"]

        self.model = PolarizationMambaSO3(input_dim=model_config["input_dim"], d_model=model_config["d_model"], n_layers=model_config["n_layers"], system=system_os).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _prepare_input(self, data):
        x = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        assert x.shape[-1] == 3, f'Data must have 3 stokes parameters'
        assert x.shape[-2] == self.window_size, f'Data must have {self.window_size} time steps'
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return x


    def predict(self, data):
        x = self._prepare_input(data)
        with torch.no_grad():
            pred = self.model(x)
        stokes_output = pred.squeeze().cpu().numpy()
        return stokes_output
