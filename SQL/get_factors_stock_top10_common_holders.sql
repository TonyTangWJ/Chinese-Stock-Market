CREATE TABLE IF NOT EXISTS factor_stock_top10_holders AS
select ts_code, ann_date, end_date, sum(hold_ratio) as sum_hold_ratio
from stock_top10_holders
where ts_code in (select ts_code from ts_code)
and ann_date >= '20100101' and ann_date <= '20241231'
group by ts_code, ann_date, end_date;