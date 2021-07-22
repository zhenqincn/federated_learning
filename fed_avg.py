import torch
from torch.utils.data import DataLoader
import os

from client import Client
from data_provider import MyDataset
from models.TwoLayerCNN import TwoLayerCNN
from server import Server
from torchvision import datasets, transforms


if __name__ == '__main__':
    # print(torch.cuda.device_count())
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    device = torch.device('cuda')
    # model = TwoLayerCNN().to(devic
    #
    # e)
    # train_data, train_label, eval_data, eval_label = load_mnist_data(if_sort=True)
    #
    # train_set = MyDataset(train_data, train_label)
    # eval_set = MyDataset(eval_data, eval_label)
    train_set = datasets.MNIST(root="./dataset/", transform=transforms.ToTensor(), train=True, download=False)
    eval_set = datasets.MNIST(root="./dataset/", transform=transforms.ToTensor(), train=False, download=False)

    train_loader = DataLoader(dataset=train_set, batch_size=600, shuffle=True)
    eval_loader = DataLoader(dataset=eval_set, batch_size=100, shuffle=True)

    server = Server(TwoLayerCNN().to(device))

    train_set_list = [item for item in train_loader]
    eval_set_list = [item for item in eval_loader]

    # model initialization
    for idx in range(len(train_set_list)):
        tmp_train_set = MyDataset(train_set_list[idx][0], train_set_list[idx][1])
        tmp_eval_set = MyDataset(eval_set_list[idx][0], eval_set_list[idx][1])
        tmp_train_loader = DataLoader(dataset=tmp_train_set, batch_size=10, shuffle=True)
        tmp_test_loader = DataLoader(dataset=tmp_eval_set, batch_size=10, shuffle=True)
        client = Client(idx, None, tmp_train_loader, tmp_test_loader, local_epoch=5, model=TwoLayerCNN().to(device),
                        cost=torch.nn.CrossEntropyLoss(), optimizer='adam')
        server.add_client(client)
    
    # model training and aggregation
    for _ in range(10):
        server.train(10, verbose=True)
        server.aggregate_model()
        server.dispatch_model()
    server.evaluate_all()

    # client = server.client_list[0]
    # client.model.load_state_dict(server.global_model.state_dict())
    # client.eval()