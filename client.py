import torch

from torch.autograd import Variable


class Client:

    def __init__(self, client_id, group_id, train_data={'x': [], 'y': []}, eval_data={'x': [], 'y': []}, local_epoch=5,
                 model=None, cost=None, optimizer=None):
        self.client_id = client_id
        self.group_id = group_id
        self.train_data = train_data
        self.eval_data = eval_data
        self.local_epoch = local_epoch
        self.model = model
        self.cost = cost
        self.optimizer = optimizer

    def train(self):
        for epoch in range(self.local_epoch):
            # train
            sum_loss = 0.0
            train_correct = 0
            for data in self.train_data:
                inputs, labels = data
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, id = torch.max(outputs.data, 1)
                sum_loss += loss.data
                train_correct += torch.sum(id == labels.data)

            print('[%d,%d] loss:%.03f' % (epoch + 1, self.local_epoch, sum_loss / len(self.train_data)))
            print('        correct:%.03f%%' % (100 * train_correct / len(self.train_data)))

    def eval(self):
        """
        :return:
            eval_correct: 评估正确的item个数
            len(self.eval_data): 参与评估的item个数
        """
        eval_correct = 0
        for data in self.eval_data:
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            outputs = self.model(inputs)
            _, idx = torch.max(outputs.data, 1)
            eval_correct += torch.sum(idx == labels.data)
        return eval_correct, len(self.eval_data)

    def get_train_data_size(self):
        return len(self.train_data)
