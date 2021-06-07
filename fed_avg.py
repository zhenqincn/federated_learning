from models.TwoLayerCNN import TwoLayerCNN
import torch


if __name__ == '__main__':
    device = torch.device('cuda')
    model = TwoLayerCNN().to(device)
    