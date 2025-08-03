import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
from utils import RiskMetrics


class RollingTrainTest:
    def __init__(self, model, Data, train_size=0.5, test_size=0.1, epochs=50, patience=5, criterion=None):
        self.model = model
        self.model_name = self.model.model_name
        self.Data = Data
        self.train_size_original = train_size
        self.train_size = train_size
        self.test_size = test_size
        self.epochs = epochs
        self.patience = patience
        self.criterion = criterion

    def info(self, predictability):
        self.predictability = predictability

        return

    def run(self):
        self.predictability = []
        self.pred_list = []
        self.pred_list_top10 = []
        self.act_list = []
        num_iterations = int((1 - self.train_size) / self.test_size)
        for _ in range(num_iterations):
            self.train_loader = self.Data.get_train_loader(train_size=self.train_size)
            self.test_loader = self.Data.get_test_loader(test_size=self.test_size)
            self.model.fit(self.train_loader, epochs=self.epochs, patience=self.patience, criterion=self.criterion)
            self.model.load_model()  # Load the best model after training
            result= round(self.model.evaluate(self.test_loader),4)
            result_train = round(self.model.evaluate(self.train_loader),4)
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
            print (f'No.{_+1} Test Predictability: {result:.4f}')
            print (f'No.{_+1} Train Predictability: {result_train:.4f}')
            # save test predictability and train predictability
            file = f'../CSV/predictability_{self.model_name}.csv'
            mode = 'a' if os.path.exists(file) else 'w'
            with open(file, mode) as f:
                if mode == 'w':
                    f.write(f'model_name,No.,test_predictability,train_predictability\n')
                if _ == 0:
                    f.write(f'{self.predictability}')
                f.write(f'{self.model_name},{_+1},{result:.4f},{result_train:.4f}\n')
            self.train_size += self.test_size
            # self.model.reset_model()
        self.pred = np.concatenate(self.pred_list, axis=0)
        self.act = np.concatenate(self.act_list, axis=0)
        self.pred_top10 = np.concatenate(self.pred_list_top10, axis=0)
        print(f"Predictability of {self.model_name}: {sum(self.predictability) / len(self.predictability):.4f}")
        file = '../CSV/predictability.csv'
        mode = 'a' if os.path.exists(file) else 'w'
        with open(file, mode) as f:
            if mode == 'w':
                f.write('model_name,predictability\n')
            f.write(f'{self.model_name},{sum(self.predictability) / len(self.predictability):.4f}\n')


    def backtest(self, trade_mode=1, data_frequency='monthly'):
        if trade_mode == 1:
            pred = self.pred[:,-1]
            act = self.act[:,-1]
            # 保存预测结果和实际值
            df = pd.DataFrame({
                'pred': pred,
                'act': act
            })
            df.to_csv(f'../CSV/predictions_{self.model_name}.csv', index=False)

            pred = np.where(pred > 0, 1, 0)
            
            period_returns = pred * act
            
            # 使用月度数据计算风险指标
            risk_metrics = RiskMetrics(data_frequency=data_frequency)
            metrics = risk_metrics.calculate_metrics(period_returns)
            
            # 保存关键指标
            self.profit_rate = metrics['mean_return']
            self.sharpe_ratio = metrics['sharpe_ratio']
            
            # 保存到CSV
            file_path = '../CSV/profit_indicators.csv'
            mode = 'a' if os.path.exists(file_path) else 'w'      
            with open(file_path, mode) as f:
                if mode == 'w':
                    f.write('model_name,mean_return,sharpe_ratio,annualized_return,annualized_volatility,win_rate\n')
                f.write(f'{self.model_name},{metrics["mean_return"]:.4f},{metrics["sharpe_ratio"]:.4f},{metrics["annualized_return"]:.4f},{metrics["annualized_volatility"]:.4f},{metrics["win_rate"]:.4f}\n')
        else:
            pass
        return
    
    '''
    def backtest_top10(self, trade_mode = 1, data_frequency='monthly'):
        if trade_mode == 1:
            pred = self.pred_top10[:,-1]
            act = self.act[:,-1]
            pred = np.where(pred > 0, 1, 0)
            
            period_returns = pred * act
            
            # 使用月度数据计算风险指标
            risk_metrics = RiskMetrics(data_frequency=data_frequency)
            metrics = risk_metrics.calculate_metrics(period_returns)
            
            # 保存关键指标
            self.profit_rate_top10 = metrics['mean_return']
            self.sharpe_ratio_top10 = metrics['sharpe_ratio']
            
            # 保存到CSV
            file_path = '../CSV/profit_indicators_top10.csv'
            mode = 'a' if os.path.exists(file_path) else 'w'      
            with open(file_path, mode) as f:
                if mode == 'w':
                    f.write('model_name,mean_return,sharpe_ratio,annualized_return,annualized_volatility,win_rate\n')
                f.write(f'{self.model_name},{metrics["mean_return"]:.4f},{metrics["sharpe_ratio"]:.4f},{metrics["annualized_return"]:.4f},{metrics["annualized_volatility"]:.4f},{metrics["win_rate"]:.4f}\n')
        else:
            pass

        return
    '''
        