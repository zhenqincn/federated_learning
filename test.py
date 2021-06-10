import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.TwoLayerCNN import TwoLayerCNN

if __name__ == '__main__':
    data_train = datasets.MNIST(root="./dataset/", transform=transforms.ToTensor(), train=True, download=False)
    data_test = datasets.MNIST(root="./dataset/", transform=transforms.ToTensor(), train=False, download=False)

    train_loader = DataLoader(dataset=data_train, batch_size=100, shuffle=True)
    test_loader = DataLoader(dataset=data_test, batch_size=100, shuffle=True)

    device = torch.device('cuda')
    model = TwoLayerCNN().to(device)
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 1
    for epoch in range(epochs):
        # train
        sum_loss = 0.0
        train_correct = 0
        for data in train_loader:
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cost(outputs, labels)
            loss.backward()
            optimizer.step()

            _, idx = torch.max(outputs.data, 1)
            sum_loss += loss.data
            train_correct += torch.sum(idx == labels.data)

        print('[%d,%d] loss:%.03f' % (epoch + 1, epochs, sum_loss / len(train_loader)))
        print('        correct:%.03f%%' % (100 * train_correct / len(data_train)))

    test_correct = 0
    for data in test_loader:
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        outputs = model(inputs)
        _, idx = torch.max(outputs.data, 1)
        test_correct += torch.sum(idx == labels.data)
    print()
    print('test correct:%.03f%%' % (100 * test_correct / len(data_test)))

    print(model.state_dict().keys())
    for k, v in model.state_dict().items():
        print(k, v)

    # for item in model.parameters():
    #     item.data = torch.zeros_like(item)
    #     print(type(item))

    # w = model.conv1.weight.data.clone()
    # print(type(w))
    # print(w)

    # model.conv1.weight.data += torch.ones_like(w)
    # print(model.conv1.weight.data)

    test_correct = 0
    for data in test_loader:
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        outputs = model(inputs)
        _, id = torch.max(outputs.data, 1)
        test_correct += torch.sum(id == labels.data)
    print()
    print('test correct:%.03f%%' % (100 * test_correct / len(data_test)))
