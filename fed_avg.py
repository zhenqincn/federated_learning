import torch
from torch.utils.data import DataLoader

from client import Client
from data_provider import load_mnist_data, MyDataset
from models.TwoLayerCNN import TwoLayerCNN
from server import Server
from torchvision import datasets, transforms

if __name__ == '__main__':
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

    train_loader = DataLoader(dataset=train_set, batch_size=600, shuffle=False)
    eval_loader = DataLoader(dataset=eval_set, batch_size=100, shuffle=False)

    server = Server(TwoLayerCNN().to(device))

    train_set_list = [item for item in train_loader]
    eval_set_list = [item for item in eval_loader]

    for idx in range(len(train_set_list)):
        tmp_train_set = MyDataset(train_set_list[idx][0], train_set_list[idx][1])
        tmp_eval_set = MyDataset(eval_set_list[idx][0], eval_set_list[idx][1])
        tmp_train_loader = DataLoader(dataset=tmp_train_set, batch_size=10, shuffle=True)
        tmp_test_loader = DataLoader(dataset=tmp_eval_set, batch_size=10, shuffle=True)
        client = Client(idx, None, tmp_train_loader, tmp_test_loader, local_epoch=5, model=TwoLayerCNN().to(device),
                        cost=torch.nn.CrossEntropyLoss(), optimizer='adam')
        server.add_client(client)
    server.client_list[0].train()
