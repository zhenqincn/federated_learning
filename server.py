import numpy as np
import torch


class Server:
    def __init__(self, global_model):
        self.global_model = global_model
        self.client_list = []

    def add_client(self, client):
        self.client_list.append(client)

    def train(self, n_samples, strategy='random'):
        """
        筛选client，然后进行训练
        :param n_samples:
        :param strategy:
        :return:
        """
        n_samples = max(n_samples, len(self.client_list))
        selected_client_ids = []

        if strategy == 'random':
            selected_client_ids.extend(np.random.choice(range(len(self.client_list)), size=n_samples, replace=False))

        for idx_client in selected_client_ids:
            self.client_list[idx_client].train()

    def aggregate_model(self, selected_client_ids):
        """
        聚合模型
        :return:
        """
        data_total = sum([self.client_list[idx_client].get_train_data_size() for idx_client in selected_client_ids])

        tmp_global_dict = self.global_model.state_dict()
        for k in self.global_model.state_dict().keys():
            tmp_global_dict[k] = torch.stack(
                [self.client_list[idx_client].model.stat_dict()[k] * self.client_list[
                    idx_client].get_train_data_size() / data_total for idx_client in selected_client_ids])
        self.global_model.load_state_dict(tmp_global_dict)

    def evaluate_all(self):
        sum_correct, sum_total = 0, 0
        for client in self.client_list:
            correct, total = client.eval()
            sum_correct += correct
            sum_total += total
        return sum_correct, sum_total
