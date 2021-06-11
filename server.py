import numpy as np
import torch


class Server:
    def __init__(self, global_model):
        self.global_model = global_model
        self.client_list = []
        self.selected_client_ids = []

    def add_client(self, client):
        self.client_list.append(client)

    def train(self, n_samples, strategy='random', verbose=False):
        """
        筛选client，然后进行训练
        :param n_samples:
        :param strategy:
        :return:
        """
        n_samples = max(n_samples, 1)
        self.selected_client_ids = []

        if strategy == 'random':
            self.selected_client_ids.extend(np.random.choice(range(len(self.client_list)), size=n_samples, replace=False))

        for idx_client in self.selected_client_ids:
            self.client_list[idx_client].train(verbose=False)

    def aggregate_model(self):
        """
        聚合模型
        :return:
        """
        data_total = sum([self.client_list[idx_client].get_train_data_size() for idx_client in self.selected_client_ids])

        tmp_global_dict = self.global_model.state_dict()
        for k in self.global_model.state_dict().keys():
            tmp_global_dict[k] = torch.stack(
                [self.client_list[idx_client].model.state_dict()[k].float() * self.client_list[
                    idx_client].get_train_data_size() / data_total for idx_client in self.selected_client_ids], 0).sum(0)
        self.global_model.load_state_dict(tmp_global_dict)
    
    def dispatch_model(self):
        global_dict = self.global_model.state_dict()
        for client in self.client_list:
            client.model.load_state_dict(global_dict)
            
    def evaluate_all(self, verbose=False):
        sum_correct, sum_total = 0, 0
        for client in self.client_list:
            correct, total = client.eval(verbose=verbose)
            sum_correct += correct
            sum_total += total
        print('evaluate all: correct:%.03f%%' % (sum_total * 100 / sum_total))
        return sum_correct, sum_total
