import tushare as ts
import pandas as pd
from sql import sql
from tqdm.auto import tqdm
import warnings
import time
import numpy as np
from datetime import datetime, timedelta
warnings.simplefilter(action='ignore', category=FutureWarning)


class prepare_data:

    def __init__(self):
        self.sql=sql()
        self.ts = ts
        self.pro = self.ts.pro_api()
        self.start_date=201501
        self.end_date=202412

    # change the start_date and end_date to the desired range
    def prepare_label(self):
        """
        Prepare the label data for the stock market.
        """
        label = self.sql.return_data(
            'SELECT * FROM a_final_label_monthly',
            database='csm'
        )
        label = label[(label['trade_date'] >= self.start_date) & (label['trade_date'] <= self.end_date)]
        label.dropna(subset=['close_return'], inplace=True)
        sql().insert_data(label, 'label_monthly')
        return
    
    # change the start_date and end_date to the desired range
    def prepare_factor_monthly(self):
        """
        Prepare the factor data for the stock market.
        """
        factor = self.sql.return_data(
            'SELECT * FROM a_final_factor_monthly',
            database='csm'
        )
        factor = factor[(factor['trade_date'] >= self.start_date) & (factor['trade_date'] <= self.end_date)]
        code_date = sql().return_data('SELECT * FROM code_date')
        code_date['trade_date'] = code_date['trade_date'].astype(int)
        factor = factor.merge(code_date, on=['ts_code', 'trade_date'], how='inner')
        sql().insert_data(factor, 'factor_monthly')
        return
    
    # change the start_date and end_date to the desired range
    def prepare_factor_daily(self):
        """
        Prepare the factor data for the stock market on a daily basis.
        """
        factor = sql().return_data(
            'SELECT * FROM a_final_factor_daily',
            database='csm'
        )

        factor = factor[(factor['trade_date'] >= 201501) & (factor['trade_date'] <= 202412)]
        factor[['volume_ratio','pb', 'pe_ttm', 'ps_ttm', 'dv_ttm', 'pcf']] = factor[['volume_ratio','pb', 'pe_ttm', 'ps_ttm', 'dv_ttm', 'pcf']].fillna(0)
        code_date = sql().return_data('SELECT * FROM code_date')
        code_date['trade_date'] = code_date['trade_date'].astype(int)
        factor = factor.merge(code_date, on=['ts_code', 'trade_date'], how='inner')
        sql().insert_data(factor, 'factor_daily')
        return

    # calculate the mean of daily factors for each stock on a monthly basis
    def prepare_factor_all(self):
        month_data = sql().return_data("select * from factor_monthly")
        day_data = sql().return_data("select * from factor_daily")
        day_data.drop(columns = ['date'], inplace = True)
        # get the mean of each column group by (ts_code and trade_date)
        day_data = day_data.groupby(['ts_code', 'trade_date']).mean().reset_index()
        # keep 2 floating point digits
        day_data = day_data.groupby(['ts_code', 'trade_date']).round(2)
        # keep 2 floating point digits
        # 'total_mv', 'turnover_rate','volume_ratio', 'pb', 'pe_ttm', 'ps_ttm', 'dv_ttm', 'pcf'
        day_data['total_mv'] = day_data['total_mv'].apply(lambda x: round(x, 2))
        day_data['turnover_rate'] = day_data['turnover_rate'].apply(lambda x: round(x, 2))
        day_data['volume_ratio'] = day_data['volume_ratio'].apply(lambda x: round(x, 2))
        day_data['pb'] = day_data['pb'].apply(lambda x: round(x, 2))
        day_data['pe_ttm'] = day_data['pe_ttm'].apply(lambda x: round(x, 2))
        day_data['ps_ttm'] = day_data['ps_ttm'].apply(lambda x: round(x, 2))
        day_data['dv_ttm'] = day_data['dv_ttm'].apply(lambda x: round(x, 2))
        day_data['pcf'] = day_data['pcf'].apply(lambda x: round(x, 2))
        # merge day_data and month_data
        data = pd.merge(month_data, day_data, on=['ts_code', 'trade_date'], how='outer')
        sql().insert_data(data, 'factor_all')
        return



class data_engineering:

    def __init__(self):
        # if cleaned data exists, load it
        try:
            self.factor = pd.read_csv('../CSV/factor_cleaned.csv')
            self.label = pd.read_csv('../CSV/label_cleaned.csv')
        except FileNotFoundError:
            print ("Cleaned data not found, loading raw data and cleaning...")
            self.factor = pd.read_csv('../CSV/factor.csv')
            self.label = pd.read_csv('../CSV/label.csv')
            self._clean_factor_data()
            self._clean_label_data()
            # save the cleaned data
            self.factor.to_csv('../CSV/factor_cleaned.csv', index=False)
            self.label.to_csv('../CSV/label_cleaned.csv', index=False)
            print("Cleaned data saved to CSV files.")

    def _clean_factor_data(self):
        """
        清理因子数据中的异常值
        """
        
        # 获取数值列
        numeric_columns = self.factor.select_dtypes(include=[np.number]).columns.tolist()
        if 'ts_code' in numeric_columns:
            numeric_columns.remove('ts_code')
        if 'trade_date' in numeric_columns:
            numeric_columns.remove('trade_date')
        
        # 处理异常值（使用3σ原则或分位数方法）
        for col in numeric_columns:
            if col in self.factor.columns:
                # 计算分位数
                Q1 = self.factor[col].quantile(0.04)
                Q3 = self.factor[col].quantile(0.96)
                
                # 将异常值替换为分位数边界值
                self.factor[col] = self.factor[col].clip(lower=Q1, upper=Q3)
        
        # 保留4位小数
        self.factor[numeric_columns] = self.factor[numeric_columns].round(4)
        return

    def _clean_label_data(self):
        """
        清理标签数据中的异常值
        """
        # 将数值除以100,去掉百分号
        self.label['highest_return'] = self.label['highest_return'] / 100
        self.label['lowest_return'] = self.label['lowest_return'] / 100
        self.label['close_return'] = self.label['close_return'] / 100
        
        # 处理收益率异常值（超出合理范围的收益率）
        if 'close_return' in self.label.columns:
            # 限制收益率在-20%到20%之间, 限制最高收益率在0%到30%之间, 限制最低收益率在-30%到0%之间
            self.label['close_return'] = self.label['close_return'].clip(lower=-0.2, upper=0.2)
            self.label['highest_return'] = self.label['highest_return'].clip(lower=0, upper=0.3)
            self.label['lowest_return'] = self.label['lowest_return'].clip(lower=-0.3, upper=0)
        
        # 收益率数据保留4位小数
        self.label['close_return'] = self.label['close_return'].round(4)
        self.label['highest_return'] = self.label['highest_return'].round(4)
        self.label['lowest_return'] = self.label['lowest_return'].round(4)
        return

    def moving_average(self, window=6):
        """
        Calculate the moving average of the factors groupby tscode.
        """
        # 获取需要计算移动平均的数值列（排除ts_code和trade_date）
        numeric_columns = self.factor.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'ts_code' in numeric_columns:
            numeric_columns.remove('ts_code')
        if 'trade_date' in numeric_columns:
            numeric_columns.remove('trade_date')
        
        # 按股票代码分组计算移动平均
        for col in numeric_columns:
            self.factor[col] = self.factor.groupby('ts_code')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
        
        # 保留4位小数
        self.factor[numeric_columns] = self.factor[numeric_columns].round(4)

        # 保存处理后的数据
        self.factor.to_csv(f'../CSV/factor_ma{window}.csv', index=False)
        
        return