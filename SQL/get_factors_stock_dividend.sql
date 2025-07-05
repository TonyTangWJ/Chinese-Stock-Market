CREATE TABLE IF NOT EXISTS factor_stock_dividend AS
select ts_code, ann_date, end_date, stk_div, cash_div_tax from stock_dividend
where ts_code in (select ts_code from ts_code)
and ann_date >= '20100101' and ann_date <= '20241231';
