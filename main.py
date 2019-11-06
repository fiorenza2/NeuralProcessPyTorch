import numpy as np
from sklearn import gaussian_process
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from model import NeuralProcess, NeuralProcessLoss
from utils import GPDataGenerator

def main():
    num_epochs = 100

    gp_data_generator = GPDataGenerator()
    x_context, x_target, y_context, y_target = gp_data_generator.create_training_set()

    neural_process = NeuralProcess(1, 1, 10, 10, 10, width=200)
    optimizer = torch.optim.Adam(neural_process.parameters(), lr=1e-3)

    neural_process.train()

    for i in range(num_epochs):
        epoch_loss = 0
        epoch_mse = 0
        for x_c, x_t, y_c, y_t in zip(x_context, x_target, y_context, y_target):
            data_context = torch.Tensor(np.concatenate((x_c.reshape(-1,1), y_c.reshape(-1,1)), 1))
            data_target = torch.Tensor(np.concatenate((x_t.reshape(-1,1), y_t.reshape(-1,1)), 1))
            optimizer.zero_grad()
            loss = NeuralProcessLoss(neural_process, data_context, data_target)
            epoch_loss += loss
            epoch_mse += nn.functional.mse_loss(neural_process(data_context, data_target[:,0]), torch.Tensor(y_t).reshape(-1,1))
            loss.backward()
            optimizer.step()
        print('Epoch:{} Loss: {:.4f} Acc: {:.4f}'.format(
                i, epoch_loss, epoch_mse))

if __name__ == "__main__":
    main()