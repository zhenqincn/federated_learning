from torch.utils.data import DataLoader

from models.TwoLayerCNN import TwoLayerCNN
import torch
import os
from data_provider import load_mnist_data, MyDataset
import numpy as np
import matplotlib.pyplot as plt
from server import Server
from client import Client

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    # device = torch.device('cuda')
    # model = TwoLayerCNN().to(device)
    train_data, train_label, eval_data, eval_label = load_mnist_data(if_sort=True)
    # server = Server(TwoLayerCNN())

    train_set = MyDataset(train_data, train_label)
    eval_set = MyDataset(eval_data, eval_label)
    train_loader = DataLoader(dataset=train_set, batch_size=600, shuffle=False)
    eval_loader = DataLoader(dataset=eval_set, batch_size=100, shuffle=False)

    for _ in range(len(train_loader)):
        pass

