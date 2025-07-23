import numpy as np
from tqdm.auto import tqdm



class RollingTrainTest:
    def __init__(self, model, Data, train_size=0.5, test_size=0.1, epochs=50, patience=5):
        self.model = model
        self.model_name = self.model.model_name
        self.Data = Data
        self.train_size = train_size
        self.test_size = test_size
        self.epochs = epochs
        self.patience = patience

    def run(self):
        self.predictability = []
        self.pred_list = []
        self.act_list = []
        num_iterations = int((1 - self.train_size) / self.test_size)
        for _ in range(num_iterations):
            self.train_loader = self.Data.get_train_loader(train_size=self.train_size)
            self.test_loader = self.Data.get_test_loader(test_size=self.test_size)
            self.model.fit(self.train_loader, epochs=self.epochs, patience=self.patience)
            result= round(self.model.evaluate(self.test_loader),4)
            label_mean = self.Data.dataloader.label_mean
            label_std = self.Data.dataloader.label_std
            pred, act = self.model.predict(self.test_loader, label_mean, label_std)
            self.pred_list.append(pred)
            self.act_list.append(act)
            self.predictability.append(result)
            print (f'No.{_+1} Predictability: {result}')
            self.train_size += self.test_size
        self.pred = np.concatenate(self.pred_list, axis=0)
        self.act = np.concatenate(self.act_list, axis=0)
        print(f"Predictability of {self.model_name}: {sum(self.predictability) / len(self.predictability)}")

    def backtest(self):
        
        return













