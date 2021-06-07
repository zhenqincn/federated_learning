class Client:

    def __init__(self, client_id, group_id, train_data={'x' : [], 'y' : []}, eval_data={'x' : [], 'y' : []}, model=None):
        self.client_id = client_id
        self.group_id = group_id
        self.train_data = train_data
        self.eval_data = eval_data
        self.model = model

    
    def train(self, num_epoch=1, batch_size=10):
        pass