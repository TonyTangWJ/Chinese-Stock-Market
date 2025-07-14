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






