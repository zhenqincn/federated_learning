import torch
from torch.utils.data import DataLoader

from client import Client
from data_provider import load_mnist_data, MyDataset
from models.TwoLayerCNN import TwoLayerCNN
from server import Server

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    # device = torch.device('cuda')
    # model = TwoLayerCNN().to(device)
    train_data, train_label, eval_data, eval_label = load_mnist_data(if_sort=True)

    train_set = MyDataset(train_data, train_label)
    eval_set = MyDataset(eval_data, eval_label)
    train_loader = DataLoader(dataset=train_set, batch_size=600, shuffle=False)
    eval_loader = DataLoader(dataset=eval_set, batch_size=100, shuffle=False)

    server = Server(TwoLayerCNN())

    train_set_list = [item for item in train_loader]
    eval_set_list = [item for item in eval_loader]

    for idx in range(len(train_set_list)):
        client = Client(idx, None, train_set_list[idx], eval_set_list[idx], local_epoch=5, model=TwoLayerCNN(),
                        cost=torch.nn.CrossEntropyLoss(), optimizer='adam')
        server.add_client(client)
    server.client_list[0].train()
