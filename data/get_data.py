# data from tushare, choice, CSMAR, and tonghuashun
import tushare as ts
import pandas as pd
from sql import sql
from tqdm.auto import tqdm
import warnings
import time
import numpy as np
from datetime import datetime, timedelta
warnings.simplefilter(action='ignore', category=FutureWarning)

class get_data:

    def __init__(self):
        self.sql=sql()
        self.ts = ts
        self.pro = self.ts.pro_api()
        self.start_date='20100101'
        self.end_date='20241231'

    # get the stock list and then store it into the database
    # data source: tushare, choice
    # without stocks listed in STAR market and Beijing Stock Exchange
    # the stocks listed before 2023-01-01 and delisted before 2016-02-01 are included
    # data research period: 2015.01.01-2024.12.31
    def get_stock_list(self):
        data_L = self.pro.stock_basic(list_status = 'L',fields='ts_code,market,area,list_date,delist_date,act_ent_type')
        data_D = self.pro.stock_basic(list_status = 'D',fields='ts_code,market,area,list_date,delist_date,act_ent_type')
        data = pd.concat([data_L,data_D],axis=0)
        data = data[~(data['delist_date']<'20160201')]
        data = data[~(data['list_date']>'20230101')]
        data.drop(columns=['delist_date','list_date'],inplace=True)
        data = data[~(data['market'] == '科创板')]
        data = data[~(data['market'] == '北交所')]
        data.loc[data['market'] == '创业板', 'market'] = 'GEM'
        data.loc[data['market'] == '主板', 'market'] = 'MainBoard'
        data.loc[data['act_ent_type'].isnull(), 'act_ent_type'] = 'others'
        data.loc[data['act_ent_type'] == '无', 'act_ent_type'] = 'others'
        data.loc[data['act_ent_type'] == '集体企业', 'act_ent_type'] = 'private'
        data.loc[data['act_ent_type'] == '外资企业', 'act_ent_type'] = 'others'
        data.loc[data['act_ent_type'] == '校办企业', 'act_ent_type'] = 'others'
        data.loc[data['act_ent_type'] == '民营企业', 'act_ent_type'] = 'private'
        data.loc[data['act_ent_type'] == '地方国企', 'act_ent_type'] = 'state'
        data.loc[data['act_ent_type'] == '中央国企', 'act_ent_type'] = 'state'
        data_area_new = pd.read_csv('../CSV/stock_area_fillna.csv',encoding='gbk')
        data_area_none = data[data['area'].isnull()]
        data.dropna(subset=['area'],inplace=True)
        data_area_none.set_index('ts_code', inplace=True)
        data_area_new.set_index('ts_code', inplace=True)
        data_area_none.update(data_area_new)
        data_area_none.reset_index(inplace=True)
        data = pd.concat([data,data_area_none],axis=0)
        self.sql.insert_data(data,'stock_list')
        return data

    # get the daily data of the stock and then store it into the database
    # data source: tushare
    def get_daily_trading_data(self):
        stock_list = self.sql.return_data('select ts_code from stock_list')['ts_code'].tolist()
        for ts_code in tqdm(stock_list, desc="Collecting daily trading data", position = 0, colour='#FA6780'):
            data = ts.pro_bar(ts_code=ts_code, 
                              adj='qfq', 
                              freq = 'D',
                              start_date=self.start_date, 
                              end_date=self.end_date,
                              fields='ts_code,trade_date,open,high,low,close,pct_chg,vol')
            data['vol'] = data['vol'].apply(lambda x: int(x * 100))
            self.sql.insert_data(data,'stock_daily_trading_data')
        return
    
    # get the weekly data of the stock and then store it into the database
    # data source: tushare
    def get_weekly_trading_data(self):
        data = self.sql.return_data('select ts_code from stock_list')
        ts_codes = data['ts_code'].tolist()

        for ts_code in tqdm(ts_codes, desc="Collecting monthly trading data", position = 0, colour='#FA6780'):
            daily_data = self.sql.return_data(f'select * from daily_trading_data where ts_code = \'{ts_code}\'')
            daily_data['trade_date'] = pd.to_datetime(daily_data['trade_date'], format='%Y%m%d')
            daily_data.set_index('trade_date', inplace=True)
            daily_data.sort_index(inplace=True)
            weekly_data = daily_data.resample('W').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'vol': 'sum'
            })
            weekly_data['pct_chg'] = weekly_data['close'].pct_change() * 100
            weekly_data['pct_chg'] = weekly_data['pct_chg'].apply(lambda x: round(x, 2) if not pd.isnull(x) else x)
            weekly_data['ts_code'] = ts_code
            weekly_data.reset_index(inplace=True)
            weekly_data['trade_date'] = weekly_data['trade_date'].dt.strftime('%Y%m%d')
            weekly_data = weekly_data[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pct_chg', 'vol']]
            weekly_data.loc[weekly_data['trade_date'] == min(weekly_data['trade_date']), 'pct_chg'] = 0
            weekly_data.dropna(inplace = True)
            self.sql.insert_data(weekly_data, 'stock_weekly_trading_data')    
        return

    

    # get the monthly data of the stock and then store it into the database
    # data source: tushare
    def get_monthly_trading_data(self):
        data = self.sql.return_data('select ts_code from stock_list')
        ts_codes = data['ts_code'].tolist()

        for ts_code in tqdm(ts_codes, desc="Collecting monthly trading data", position = 0, colour='#FA6780'):
            daily_data = self.sql.return_data(f'select * from daily_trading_data where ts_code = \'{ts_code}\'')
            daily_data['trade_date'] = pd.to_datetime(daily_data['trade_date'], format='%Y%m%d')
            daily_data.set_index('trade_date', inplace=True)
            daily_data.sort_index(inplace=True)
            monthly_data = daily_data.resample('M').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'vol': 'sum'
            })
            monthly_data['pct_chg'] = monthly_data['close'].pct_change() * 100
            monthly_data['pct_chg'] = monthly_data['pct_chg'].apply(lambda x: round(x, 2) if not pd.isnull(x) else x)
            monthly_data['ts_code'] = ts_code
            monthly_data.reset_index(inplace=True)
            monthly_data['trade_date'] = monthly_data['trade_date'].dt.strftime('%Y%m%d')
            monthly_data = monthly_data[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pct_chg', 'vol']]
            monthly_data.loc[monthly_data['trade_date'] == min(monthly_data['trade_date']), 'pct_chg'] = 0
            monthly_data.dropna(inplace = True)
            self.sql.insert_data(monthly_data, 'stock_monthly_trading_data')  
        return
    
    
    # get the balance sheet data of the stock and then store it into the database
    # data source: tushare
    def get_balance_sheet(self):
        stock_list = self.sql.return_data('select ts_code from stock_list')['ts_code'].tolist()
        for ts_code in tqdm(stock_list, desc="Collecting balance sheet data", position = 0, colour='#FA6780'):
            data = self.pro.balancesheet(ts_code=ts_code, 
                                         start_date=self.start_date, 
                                         end_date=self.end_date, 
                                         report_type = '1',
                                         fields='')
            data = data.sort_values(by=['ts_code', 'f_ann_date', 'end_date','update_flag'], ascending=[True, True, True, False])
            data = data.drop_duplicates(subset=['ts_code', 'f_ann_date','end_date'], keep='first')
            data.drop(columns=['ann_date', 'update_flag', 'report_type','end_type'], inplace=True)
            data = data.applymap(lambda x: None if pd.isnull(x) else str(round(x, 2)) if isinstance(x, (int, float)) else x)
            self.sql.insert_data(data,'repo_balance_sheet')
        return

    # get the income statement data of the stock and then store it into the database
    # data source: tushare
    def get_income_statement(self):
        stock_list = self.sql.return_data('select ts_code from stock_list')['ts_code'].tolist()
        for ts_code in tqdm(stock_list, desc="Collecting income statement data", position = 0, colour='#FA6780'):
            data = self.pro.income(ts_code=ts_code, 
                                  start_date=self.start_date, 
                                  end_date=self.end_date, 
                                  report_type = '1',
                                  fields='')
            data = data.sort_values(by=['ts_code', 'f_ann_date', 'end_date','update_flag'], ascending=[True, True, True, False])
            data = data.drop_duplicates(subset=['ts_code', 'f_ann_date','end_date'], keep='first')
            data.drop(columns=['ann_date', 'update_flag', 'report_type','end_type'], inplace=True)
            data = data.applymap(lambda x: None if pd.isnull(x) else str(round(x, 2)) if isinstance(x, (int, float)) else x)
            self.sql.insert_data(data,'repo_income_statement')
        return
    
    # get the cashflow statement data of the stock and then store it into the database
    # data source: tushare
    def get_cashflow_statement(self):
        stock_list = self.sql.return_data('select ts_code from stock_list')['ts_code'].tolist()
        for ts_code in tqdm(stock_list, desc="Collecting cashflow statement data", position = 0, colour='#FA6780'):
            data = self.pro.cashflow(ts_code=ts_code, 
                                   start_date=self.start_date, 
                                   end_date=self.end_date, 
                                   report_type = '1',
                                   fields='')
            data = data.sort_values(by=['ts_code', 'f_ann_date', 'end_date','update_flag'], ascending=[True, True, True, False])
            data = data.drop_duplicates(subset=['ts_code', 'f_ann_date','end_date'], keep='first')
            data.drop(columns=['ann_date', 'update_flag', 'report_type','end_type'], inplace=True)
            data = data.applymap(lambda x: None if pd.isnull(x) else str(round(x, 2)) if isinstance(x, (int, float)) else x)
            self.sql.insert_data(data,'repo_cashflow_statement')
        return
    
    # get the financial indicator data of the stock and then store it into the database
    # data source: tushare
    def get_financial_indicator(self):
        stock_list = self.sql.return_data('select ts_code from stock_list')['ts_code'].tolist()
        for ts_code in tqdm(stock_list, desc="Collecting financial indicator data", position = 0, colour='#FA6780'):
            data = self.pro.fina_indicator(ts_code=ts_code, 
                            start_date=self.start_date, 
                            end_date=self.end_date, 
                            fields='ts_code,ann_date,end_date,eps,dt_eps,total_revenue_ps,revenue_ps,capital_rese_ps,surplus_rese_ps,undist_profit_ps,extra_item,profit_dedt,gross_margin,current_ratio,quick_ratio,cash_ratio,invturn_days,arturn_days,inv_turn,ar_turn,ca_turn,fa_turn,assets_turn,op_income,valuechange_income,interst_income,daa,ebit,ebitda,fcff,fcfe,current_exint,noncurrent_exint,interestdebt,netdebt,tangible_asset,working_capital,networking_capital,invest_capital,retained_earnings,diluted2_eps,bps,ocfps,retainedps,cfps,ebit_ps,fcff_ps,fcfe_ps,netprofit_margin,grossprofit_margin,cogs_of_sales,expense_of_sales,profit_to_gr,saleexp_to_gr,adminexp_of_gr,finaexp_of_gr,impai_ttm,gc_of_gr,op_of_gr,ebit_of_gr,roe,roe_waa,roe_dt,roa,npta,roic,roe_yearly,roa2_yearly,roe_avg,opincome_of_ebt,investincome_of_ebt,n_op_profit_of_ebt,tax_to_ebt,dtprofit_to_profit,salescash_to_or,ocf_to_or,ocf_to_opincome,capitalized_to_da,debt_to_assets,assets_to_eqt,dp_assets_to_eqt,ca_to_assets,nca_to_assets,tbassets_to_totalassets,int_to_talcap,eqt_to_talcapital,currentdebt_to_debt,longdeb_to_debt,ocf_to_shortdebt,debt_to_eqt,eqt_to_debt,eqt_to_interestdebt,tangibleasset_to_debt,tangasset_to_intdebt,tangibleasset_to_netdebt,ocf_to_debt,ocf_to_interestdebt,ocf_to_netdebt,ebit_to_interest,longdebt_to_workingcapital,ebitda_to_debt,turn_days,roa_yearly,roa_dp,fixed_assets,profit_prefin_exp,non_op_profit,op_to_ebt,nop_to_ebt,ocf_to_profit,cash_to_liqdebt,cash_to_liqdebt_withinterest,op_to_liqdebt,op_to_debt,roic_yearly,total_fa_trun,profit_to_op,q_opincome,q_investincome,q_dtprofit,q_eps,q_netprofit_margin,q_gsprofit_margin,q_exp_to_sales,q_profit_to_gr,q_saleexp_to_gr,q_adminexp_to_gr,q_finaexp_to_gr,q_impair_to_gr_ttm,q_gc_to_gr,q_op_to_gr,q_roe,q_dt_roe,q_npta,q_opincome_to_ebt,q_investincome_to_ebt,q_dtprofit_to_profit,q_salescash_to_or,q_ocf_to_sales,q_ocf_to_or,basic_eps_yoy,dt_eps_yoy,cfps_yoy,op_yoy,ebt_yoy,netprofit_yoy,dt_netprofit_yoy,ocf_yoy,roe_yoy,bps_yoy,assets_yoy,eqt_yoy,tr_yoy,or_yoy,q_gr_yoy,q_gr_qoq,q_sales_yoy,q_sales_qoq,q_op_yoy,q_op_qoq,q_profit_yoy,q_profit_qoq,q_netprofit_yoy,q_netprofit_qoq,equity_yoy,rd_exp,update_flag')
            data = data.sort_values(by=['ts_code', 'ann_date','end_date','update_flag'], ascending=[True, True, True, False])
            data = data.drop_duplicates(subset=['ts_code', 'ann_date','end_date'], keep='first')
            data.drop(columns=['update_flag'], inplace=True)
            data = data.applymap(lambda x: None if pd.isnull(x) else str(round(x, 2)) if isinstance(x, (int, float)) else x)
            data = data.dropna(subset=['ann_date'])
            self.sql.insert_data(data,'repo_financial_indicator')
        return
    
    # get the stock index data of the stock and then store it into the database
    # data source: tushare, tonghuashun
    def get_stock_index(self):
        stock_list = self.sql.return_data('select ts_code from stock_list')['ts_code'].tolist()
        df = self.pro.ths_index(fields = 'ts_code, name, list_date, type')
        for con_code in tqdm(stock_list, desc="Collecting stock index data", position = 0, colour='#FA6780'):
            df2 = self.pro.ths_member(con_code = con_code, fields = 'con_code, ts_code')
            data = pd.merge(df2,df,on = 'ts_code', how = 'left')
            data.rename(columns={'ts_code':'index_code','con_code':'ts_code'}, inplace = True)
            data = data.applymap(lambda x: None if pd.isnull(x) else str(x) if isinstance(x, (int, float)) else x)
            self.sql.insert_data(data,'stock_index')
        return
    
    # get the audit opinion data of the stock and then store it into the database
    # data source: tushare
    def get_audit_opinion(self):
        stock_list = self.sql.return_data('select ts_code from stock_list')['ts_code'].tolist()
        for ts_code in tqdm(stock_list, desc="Collecting audit opinion data", position = 0, colour='#FA6780'):
            data = self.pro.fina_audit(ts_code=ts_code, 
                                     start_date=self.start_date, 
                                     end_date=self.end_date, 
                                     fields='ts_code,ann_date,end_date,audit_result')
            data.dropna(inplace = True)
            self.sql.insert_data(data,'repo_audit_opinion')
        return

    # get the stock name histories data and then stock it into the database
    # data source: tushare
    def get_stock_name_histories(self):
        stock_list = self.sql.return_data('select ts_code from stock_list')['ts_code'].tolist()
        for ts_code in tqdm(stock_list, desc="Collecting stock name history data", position = 0, colour='#FA6780'):
            data = self.pro.namechange(ts_code = ts_code, fields = 'ts_code, name, start_date, end_date, change_reason')
            data.drop_duplicates(subset=['ts_code', 'name', 'start_date'], inplace = True)
            self.sql.insert_data(data,'stock_name_histories')
        return
    
    # get the index daily trading data and then store it into the database
    # data source: tushare
    def get_index_daily_trading_data(self):
        index_list = ['000001.SH','399001.SZ','399006.SZ']
        for ts_code in tqdm(index_list, desc="Collecting index daily trading data", position = 0, colour='#FA6780'):
            data = self.ts.pro_bar(ts_code=ts_code, 
                                    asset = 'I',
                                    freq = 'D',
                                    start_date=self.start_date, 
                                    end_date=self.end_date,
                                    fields = 'ts_code, trade_date, open, high, low, close, pct_chg, vol')
            data['vol'] = data['vol'].apply(lambda x: int(x * 100))
            self.sql.insert_data(data,'index_daily_trading_data')
        return
    
    # get the index weekly trading data and then store it into the database
    # data source: tushare
    def get_index_weekly_trading_data(self):
        index_list = ['000001.SH','399001.SZ','399006.SZ']
        for ts_code in tqdm(index_list, desc="Collecting index weekly trading data", position = 0, colour='#FA6780'):
            data = self.pro.index_weekly(ts_code=ts_code, 
                                         start_date=self.start_date, 
                                         end_date='20250105', 
                                         fields='ts_code, trade_date, open, high, low, close, pct_chg, vol')
            data['vol'] = data['vol'].apply(lambda x: int(x * 100))
            self.sql.insert_data(data, 'index_weekly_trading_data')  
        return
    
    # get the index monthly trading data and then store it into the database
    # data source: tushare
    def get_index_monthly_trading_data(self):
        index_list = ['000001.SH','399001.SZ','399006.SZ']
        for ts_code in tqdm(index_list, desc="Collecting index monthly trading data", position = 0, colour='#FA6780'):
            data = self.pro.index_monthly(ts_code=ts_code, 
                                         start_date=self.start_date, 
                                         end_date='20250105', 
                                         fields='ts_code, trade_date, open, high, low, close, pct_chg, vol')
            data['vol'] = data['vol'].apply(lambda x: int(x * 100))
            self.sql.insert_data(data, 'index_monthly_trading_data')  
        return
    
    # get the market weekly trading data and then store it into the database
    # data source: CSMAR
    def get_market_weekly_trading_data(self):
        data = pd.read_csv('../CSV/weekly_market_trading.csv')
        data = data[(data['Markettype'] == 1) | (data['Markettype'] == 4) | (data['Markettype'] == 16)]
        data.loc[data['Markettype'] == 1, 'Markettype'] = 'SSE'
        data.loc[data['Markettype'] == 4, 'Markettype'] = 'SZSE'
        data.loc[data['Markettype'] == 16, 'Markettype'] = 'GEM'
        data['turnover_ratio'] = data['Wnvaltrdtl'] / data['Wmvosd'] /10
        data['turnover_ratio'] = data['turnover_ratio'].apply(lambda x: round(x, 2))
        data.drop(columns = [ 'Wnvaltrdtl', 'Wmvosd'], inplace = True)
        data.dropna(inplace = True)
        data['Wretmdeq'] = data['Wretmdeq'].apply(lambda x: round(x*100, 2))
        data['Wretmdos'] = data['Wretmdos'].apply(lambda x: round(x*100, 2))
        data['Wretmdtl'] = data['Wretmdtl'].apply(lambda x: round(x*100, 2))
        
        def week_to_date(year_week):
            year, week = map(int, year_week.split('-'))
            try:
                dt = datetime.strptime(f"{year} {week} 1", "%G %V %u")
                dt_sunday = dt + timedelta(days=6)
                return dt_sunday.strftime('%Y%m%d')
            except ValueError:
                last_sunday = datetime(year, 12, 31) - timedelta(days=datetime(year, 12, 31).weekday())
                return last_sunday.strftime('%Y%m%d')
        
        data['Trdwnt'] = data['Trdwnt'].apply(week_to_date)
        data.rename({'Markettype': 'market_pro', 
                    'Trdwnt': 'trade_date',
                    'Ndaytrd': 'trade_days',
                    'Wnshrtrdtl': 'trade_share_number',
                    'Wretmdeq': 'weekly_return_eq',
                    'Wretmdos': 'weekly_return_cv',
                    'Wretmdtl': 'weekly_return_tv',
                    'Wnstkcal': 'number_of_stocks',
                    'Wmvttl': 'weekly_market_cap'}, axis=1, inplace=True)

        self.sql.insert_data(data, 'market_weekly_trading_data')
        return
    
    # get stock daily indicator data and then store it into the database
    # data source: tushare
    def get_stock_daily_indicator(self):
        stock_list = self.sql.return_data('select ts_code from stock_list')['ts_code'].tolist()
        for ts_code in tqdm(stock_list, desc="Collecting stock daily indicator data", position = 0, colour='#FA6780'):
            data = self.pro.daily_basic(ts_code=ts_code, 
                                        start_date=self.start_date, 
                                        end_date=self.end_date,
                                        fields='ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv')
            data = data.applymap(lambda x: None if pd.isnull(x) else str(x) if isinstance(x, (int, float)) else x)
            self.sql.insert_data(data,'stock_daily_indicator')
        return
    
    # get stock dividend data and then store it into the database
    # data source: tushare
    def get_stock_dividend(self):
        stock_list = self.sql.return_data('select ts_code from stock_list')["ts_code"].tolist()
        for ts_code in tqdm(stock_list, desc="Collecting stock dividend data", position=0, colour="#FA6780"):
            fields = (
                "ts_code,imp_ann_date,end_date,div_proc,stk_div,stk_bo_rate,"
                "stk_co_rate,cash_div,cash_div_tax,base_share"
            )
            data = self.pro.dividend(ts_code=ts_code, fields=fields)
            data.dropna(subset=['imp_ann_date', 'end_date'], inplace=True)
            data.rename(columns={'imp_ann_date': 'ann_date'}, inplace = True)
            data = data.applymap(lambda x: None if pd.isnull(x) else str(x) if isinstance(x, (int, float)) else x)
            self.sql.insert_data(data, "stock_dividend")
        return data
    
    # get stock top_10 stock holders and then store it into the database
    # data source: tushare
    def get_stock_top10_holders(self):
        stock_list = self.sql.return_data('select ts_code from stock_list')["ts_code"].tolist()
        for ts_code in tqdm(stock_list, desc="Collecting stock top 10 holders data", position=0, colour="#FA6780"):
            fields = (
                "ts_code,ann_date,end_date,holder_name,hold_amount,hold_ratio,hold_float_ratio,hold_change,holder_type"
            )
            data = self.pro.top10_holders(ts_code=ts_code, fields=fields)
            data = data.applymap(lambda x: None if pd.isnull(x) else str(x) if isinstance(x, (int, float)) else x)
            self.sql.insert_data(data, "stock_top10_holders")
        return

    # get stock top_10 common stock holders and then store it into the database
    # data source: tushare
    def get_stock_top10_common_holders(self):
        stock_list = self.sql.return_data('select ts_code from stock_list')["ts_code"].tolist()
        for ts_code in tqdm(stock_list, desc="Collecting stock top 10 common holders data", position=0, colour="#FA6780"):
            fields = (
                "ts_code,ann_date,end_date,holder_name,hold_amount,hold_ratio,hold_float_ratio,hold_change,holder_type"
            )
            data = self.pro.top10_floatholders(ts_code=ts_code, fields=fields)
            data = data.applymap(lambda x: None if pd.isnull(x) else str(x) if isinstance(x, (int, float)) else x)
            self.sql.insert_data(data, "stock_top10_common_holders")
        return
    
    # calculate the pcf
    # data source: tushare
    def get_stock_pcf(self):
        cashflow_data = sql().return_data('select ts_code, f_ann_date, end_date, n_cashflow_act from repo_cashflow_statement')
        total_mv = sql().return_data('select ts_code, trade_date, total_mv from stock_daily_indicator')
        total_mv['t_trade_date'] = pd.to_datetime(total_mv['trade_date']).dt.date
        total_mv['quarter'] = pd.to_datetime(total_mv['t_trade_date']).dt.to_period('Q')
        cashflow_data['end_date'] = pd.to_datetime(cashflow_data['end_date'].astype(str)).dt.date
        cashflow_data['quarter'] = pd.to_datetime(cashflow_data['end_date']).dt.to_period('Q')
        cashflow_data.drop(columns=['f_ann_date','end_date'], inplace=True)
        data = pd.merge(cashflow_data, total_mv, on=['ts_code', 'quarter'], how='outer')
        data.drop(columns=['t_trade_date','quarter'], inplace=True)
        data['n_cashflow_act'] = data['n_cashflow_act'].replace(0, np.nan)
        data['pcf'] = (data['total_mv'] * 10000) / data['n_cashflow_act']
        data['pcf'] = data['pcf'].apply(lambda x:round(x, 2) if pd.notnull(x) else np.nan)
        data.drop(columns = ['total_mv', 'n_cashflow_act'], inplace=True)
        data.dropna(subset=['trade_date'], inplace=True)
        data.replace(np.nan, 0, inplace=True)
        data.drop_duplicates(subset=['ts_code', 'trade_date'], inplace=True)
        sql().insert_data(data,'stock_daily_pcf')
        return
    

    # collect quarterly factors
    # data source: tushare
    def get_quarterly_factors(self):
        indicator = sql().return_data('select * from factor_repo_financial_indicator')
        hold_ratio = sql().return_data('select * from factor_stock_top10_holders')
        hold_ratio['end_date'] = hold_ratio['end_date'].astype('int')
        hold_ratio.drop(columns = ['ann_date'], inplace = True)
        structure  =  sql().return_data('select * from factor_own_structure')
        operatin = sql().return_data('select * from factor_own_operatin')
        operatin.drop(columns = ['ann_date'], inplace = True)
        profit = sql().return_data('select * from factor_own_profit')
        profit.drop(columns = ['f_ann_date'], inplace = True)
        growth = sql().return_data('select * from factor_own_growth')
        growth.drop(columns = ['f_ann_date'], inplace = True)
        cash =sql().return_data('select * from factor_own_cash')
        cash.drop(columns = ['f_ann_date'], inplace = True)
        data = pd.merge(indicator, hold_ratio, how = 'left', on = ['ts_code', 'end_date'])
        data = pd.merge(data, structure, how = 'left', on = ['ts_code', 'end_date'])
        data = pd.merge(data, profit, how = 'left', on = ['ts_code', 'end_date'])
        data = pd.merge(data, operatin, how = 'left', on = ['ts_code', 'end_date'])
        data = pd.merge(data, growth, how = 'left', on = ['ts_code', 'end_date'])
        data = pd.merge(data, cash, how = 'left', on = ['ts_code', 'end_date'])
        data.drop_duplicates(subset=['ts_code', 'end_date'], inplace = True)
        data.replace(np.nan, 0, inplace=True)
        sql().insert_data(data, 'factor_quarter')
        return
