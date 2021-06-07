from models.TwoLayerCNN import TwoLayerCNN
import torch
import os
from data_provision import load_mnist_data
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    # device = torch.device('cuda')
    # model = TwoLayerCNN().to(device)
    train_data, train_label, eval_data, eval_label = load_mnist_data(if_sort=True)
    