import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
from utils import RiskMetrics
import math
import time


class RollingTrainTest:
    def __init__(self, model, Data, train_size=0.5, test_size=0.1, epochs=50, patience=5, criterion=None, count = 0):
        self.model = model
        self.model_name = self.model.model_name
        self.Data = Data
        self.train_size_original = train_size
        self.train_size = train_size
        self.test_size = test_size
        self.epochs = epochs
        self.patience = patience
        self.criterion = criterion
        self.count = count

    def info(self, predictability_name):
        self.predictability_name = predictability_name
        return

    def run(self):
        self.predictability = []
        self.pred_list = []
        self.pred_list_top10 = []
        self.act_list = []
        self.num_iterations = math.ceil((1 - self.train_size) / self.test_size)
        print (f'{self.model_name} will run {self.num_iterations} iterations.')
        all_start_time = time.time()
        for _ in range(self.num_iterations):
            start_time = time.time()
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
            # store the actual values
            self.act_list.append(act)
            self.predictability.append(result)
            # return the spent seconds
            spent_time = round(time.time() - start_time, 2)
            print (f'No.{_+1} Test Predictability: {result:.4f}')
            print (f'No.{_+1} Train Predictability: {result_train:.4f}')
            # save test predictability and train predictability
            file = f'../CSV_final/profit/predictability_details/{self.model_name}.csv'
            mode = 'a' if os.path.exists(file) else 'w'
            with open(file, mode) as f:
                if mode == 'w':
                    f.write(f'model_name,No.,test_predictability,train_predictability,spent_time\n')
                if _ == 0:
                    f.write(f'{self.predictability_name}\n')
                f.write(f'{self.model_name},{_+1},{result:.4f},{result_train:.4f},{spent_time:.2f}s\n')
            self.train_size += self.test_size
            # self.model.reset_model()
        self.pred = np.concatenate(self.pred_list, axis=0)
        self.act = np.concatenate(self.act_list, axis=0)
        all_spent_time = round(time.time() - all_start_time, 2)
        print(f"Predictability of {self.model_name}: {sum(self.predictability) / len(self.predictability):.4f}")
        file = '../CSV_final/predictability.csv'
        mode = 'a' if os.path.exists(file) else 'w'
        with open(file, mode) as f:
            if mode == 'w':
                f.write('Time,model_name,predictability,all_spent_time\n')
            if self.count == 0:
                f.write(f'{self.predictability_name}\n')
            f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())},{self.model_name},{sum(self.predictability) / len(self.predictability):.4f},{all_spent_time:.2f}s\n')


    def backtest(self, trade_mode=1, data_frequency='monthly'):

        if trade_mode == 1:
            pred = self.pred[:,-1]
            act = self.act[:,-1]
            pred = np.where(pred > 0, 1, 0)
            all_returns = pred * act

            num = math.ceil(len(all_returns) /12 /self.num_iterations)

            # 按顺序提取num个return数据,作为一个月的收益率
            # 取平均值时忽略0
            period_returns = []
            for i in range(0, len(all_returns), num):
                period_data = all_returns[i:i+num]
                non_zero_data = period_data[period_data != 0]
                if len(non_zero_data) > 0:
                    period_returns.append(np.mean(non_zero_data))
                else:
                    period_returns.append(0)
            # 保留4位小数
            period_returns = [round(x, 4) for x in period_returns]
            period_returns = np.array(period_returns)
            # 计算累计收益率
            cum_returns = np.cumprod(1 + period_returns/100) - 1
            cum_returns = [round(x*100, 4) for x in cum_returns]

            # 根据period_returns的数量分配月份,最后一个月是2024年12月
            months = pd.date_range(end='2024-12-31', periods=len(period_returns), freq='M')
            
            # save to dataframe
            df = pd.DataFrame(
                {'Month': months,
                'returns': period_returns,
                'cum_returns': cum_returns}
            )

            file = f'../CSV_final/profit/profit_details/{self.model_name}.csv'
            mode = 'a' if os.path.exists(file) else 'w'
            with open(file, mode, encoding='utf-8', newline="") as f:
                if mode == 'w':
                    # 如果是新文件，先写入描述信息
                    f.write('Month,returns,cum_returns\n')
                    f.write(f'{self.predictability_name}\n')
                    # 写入数据，不包含表头，去掉最后的换行符
                    csv_content = df.to_csv(index=False, header=False)
                    f.write(csv_content)
                else:
                    f.write(f'{self.predictability_name}\n')
                    csv_content = df.to_csv(index=False, header=False)
                    f.write(csv_content)

            # 使用月度数据计算风险指标
            risk_metrics = RiskMetrics(risk_free_rate=0.01, data_frequency=data_frequency)
            metrics = risk_metrics.calculate_metrics(period_returns)
            
            # 保存到CSV
            file_path = '../CSV_final/profit.csv'
            mode = 'a' if os.path.exists(file_path) else 'w'      
            with open(file_path, mode) as f:
                if mode == 'w':
                    f.write('Time,model_name,mean_return,sharpe_ratio,annualized_return,annualized_volatility,max_drawdown,win_rate\n')
                if self.count == 0:
                    f.write(f'{self.predictability_name}\n')
                f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())},{self.model_name},{metrics["mean_return"]:.4f},{metrics["sharpe_ratio"]:.4f},{metrics["annualized_return"]:.4f},{metrics["annualized_volatility"]:.4f},{metrics["max_drawdown"]:.4f},{metrics["win_rate"]:.4f}\n')
            
        elif trade_mode == 2:
            pred_high = self.pred[:,-3]
            pred_high = np.where(pred_high > 0, pred_high, 0)
            act_high = self.act[:,-3]
            close = self.act[:,-1]
            # 如果预测的最高值小于实际的最高值,取预测的最高值;反之,则去close
            all_returns = np.where(pred_high <= act_high, pred_high, close)

            num = math.ceil(len(all_returns) /12 /self.num_iterations)

            # 按顺序提取num个return数据,作为一个月的收益率
            # 取平均值时忽略0
            period_returns = []
            for i in range(0, len(all_returns), num):
                period_data = all_returns[i:i+num]
                non_zero_data = period_data[period_data != 0]
                if len(non_zero_data) > 0:
                    period_returns.append(np.mean(non_zero_data))
                else:
                    period_returns.append(0)
            # 保留4位小数
            period_returns = [round(x, 4) for x in period_returns]
            period_returns = np.array(period_returns)
            # 计算累计收益率
            cum_returns = np.cumprod(1 + period_returns/100) - 1
            cum_returns = [round(x*100, 4) for x in cum_returns]

            # 根据period_returns的数量分配月份,最后一个月是2024年12月
            months = pd.date_range(end='2024-12-31', periods=len(period_returns), freq='M')
            
            # save to dataframe
            df = pd.DataFrame(
                {'Month': months,
                'returns': period_returns,
                'cum_returns': cum_returns}
            )
            os.makedirs('CSV_final', exist_ok=True)
            file = f'../CSV_final/profit/profit_details/{self.model_name}.csv'
            mode = 'a' if os.path.exists(file) else 'w'
            with open(file, mode, encoding='utf-8', newline="") as f:
                if mode == 'w':
                    # 如果是新文件，先写入描述信息
                    f.write('Month,returns,cum_returns\n')
                    f.write(f'{self.predictability_name}\n')
                    # 写入数据，不包含表头，去掉最后的换行符
                    csv_content = df.to_csv(index=False, header=False)
                    f.write(csv_content)
                else:
                    f.write(f'{self.predictability_name}\n')
                    csv_content = df.to_csv(index=False, header=False)
                    f.write(csv_content)

            # 使用月度数据计算风险指标
            risk_metrics = RiskMetrics(risk_free_rate=0.01, data_frequency=data_frequency)
            metrics = risk_metrics.calculate_metrics(period_returns)
            
            # 保存到CSV
            os.makedirs('CSV_final', exist_ok=True)
            file_path = '../CSV_final/profit.csv'
            mode = 'a' if os.path.exists(file_path) else 'w'      
            with open(file_path, mode) as f:
                if mode == 'w':
                    f.write('Time,model_name,mean_return,sharpe_ratio,annualized_return,annualized_volatility,max_drawdown,win_rate\n')
                if self.count == 0:
                    f.write(f'{self.predictability_name}\n')
                f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())},{self.model_name},{metrics["mean_return"]:.4f},{metrics["sharpe_ratio"]:.4f},{metrics["annualized_return"]:.4f},{metrics["annualized_volatility"]:.4f},{metrics["max_drawdown"]:.4f},{metrics["win_rate"]:.4f}\n')
        
        self.Return = round(metrics["mean_return"],4)
        self.SR = round(metrics["sharpe_ratio"],4)
        return 
    
        