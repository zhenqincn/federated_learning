import torch

from torch.autograd import Variable


class Client:

    def __init__(self, client_id, group_id, train_data, eval_data, local_epoch=5, model=None, cost=None,
                 optimizer=None):
        self.client_id = client_id
        self.group_id = group_id
        self.train_data = train_data
        self.eval_data = eval_data
        self.local_epoch = local_epoch
        self.model = model
        self.cost = cost

        self.optimizer = torch.optim.Adam(self.model.parameters())

        self._train_data_length = 0
        self._eval_data_length = 0
        for inputs, _ in self.train_data:
            self._train_data_length += len(inputs)
        for inputs, _ in self.eval_data:
            self._eval_data_length += len(inputs)
            

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

                _, idx = torch.max(outputs.data, 1)
                sum_loss += loss.data
                train_correct += torch.sum(idx == labels.data)

            print('client %d: [%d/%d] loss:%.03f    correct:%.03f%%' % (self.client_id, epoch + 1, self.local_epoch, sum_loss / self._train_data_length, 100 * train_correct / self._train_data_length))
        print()

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
        return eval_correct, self._eval_data_length

    def get_train_data_size(self):
        return self._train_data_length

    def get_eval_data_size(self):
        return self._eval_data_length
