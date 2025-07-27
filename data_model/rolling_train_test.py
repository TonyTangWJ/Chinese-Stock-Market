import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os


class RollingTrainTest:
    def __init__(self, model, Data, train_size=0.5, test_size=0.1, epochs=50, patience=5):
        self.model = model
        self.model_name = self.model.model_name
        self.Data = Data
        self.train_size_original = train_size
        self.train_size = train_size
        self.test_size = test_size
        self.epochs = epochs
        self.patience = patience

    def run(self):
        self.predictability = []
        self.pred_list = []
        self.pred_list_top10 = []
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
            # store the predictions
            self.pred_list.append(pred)
            # keep the top 10 and bottom 10 predictions, change the others to 0, keep the order
            long_indices = np.argsort(pred)[-10:]
            short_indices = np.argsort(pred)[:10]
            pred_top10 = np.zeros_like(pred)
            pred_top10[long_indices] = pred[long_indices]
            pred_top10[short_indices] = pred[short_indices]
            self.pred_list_top10.append(pred_top10)
            # store the actual values
            self.act_list.append(act)
            self.predictability.append(result)
            print (f'No.{_+1} Predictability: {result:.4f}')
            self.train_size += self.test_size
        self.pred = np.concatenate(self.pred_list, axis=0)
        self.act = np.concatenate(self.act_list, axis=0)
        self.pred_top10 = np.concatenate(self.pred_list_top10, axis=0)
        print(f"Predictability of {self.model_name}: {sum(self.predictability) / len(self.predictability):.4f}")
        file = f'../CSV/predictability.csv'
        mode = 'a' if os.path.exists(file) else 'w'
        with open(file, mode) as f:
            if mode == 'w':
                f.write('model_name,predictability\n')
            f.write(f'{self.model_name},{sum(self.predictability) / len(self.predictability):.4f}\n')

    def backtest(self, trade_mode = 1):
        if trade_mode == 1:
            pred = self.pred[:,-1]
            act = self.act[:,-1]
            pred = np.where(pred > 0, 1, -1)
            self.profit_rate = np.mean(pred * act)/(10*(1-self.train_size_original))
            # check if the file exists and write the profit rate
            file_path = '../CSV/profit_rate.csv'
            mode = 'a' if os.path.exists(file_path) else 'w'      
            with open(file_path, mode) as f:
                if mode == 'w':
                    f.write('model_name,profit_rate\n')
                f.write(f'{self.model_name},{self.profit_rate:.4f}\n')
            print(f"Total Profit rate of {self.model_name}: {self.profit_rate:.4f}")
        else:
            pass
        return
    
    def backtest_top10(self, trade_mode = 1):
        if trade_mode == 1:
            pred = self.pred_top10[:,-1]
            act = self.act[:,-1]
            pred = np.where(pred > 0, 1, -1)
            self.profit_rate_top10 = np.mean(pred * act)/(10*(1-self.train_size_original))
            # check if the file exists and write the profit rate
            file_path = '../CSV/profit_rate_top10.csv'
            mode = 'a' if os.path.exists(file_path) else 'w'      
            with open(file_path, mode) as f:
                if mode == 'w':
                    f.write('model_name,profit_rate_top10\n')
                f.write(f'{self.model_name},{self.profit_rate_top10:.4f}\n')
            print(f"Top10 Profit rate of {self.model_name} top 10: {self.profit_rate_top10:.4f}")
        else:
            pass
        return