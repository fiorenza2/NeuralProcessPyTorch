import numpy as np
from sklearn import gaussian_process as gp
import torch
import torch.nn as nn
from .model import *

def main():
    f = gp.GaussianProcessRegressor()
    x = np.linspace(-1, 1).reshape(-1, 1)
    y = f.sample_y(x, 20)

    np = NeuralProcess()
    loss = NeuralProcessLoss(np)