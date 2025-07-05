CREATE TABLE IF NOT EXISTS factor_stock_daily_indicator AS
select ts_code, trade_date, total_mv, turnover_rate, volume_ratio, pb, pe_ttm, ps_ttm, dv_ttm from stock_daily_indicator
where ts_code in (select ts_code from ts_code);
