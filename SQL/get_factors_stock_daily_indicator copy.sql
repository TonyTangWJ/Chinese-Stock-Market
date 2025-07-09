CREATE TABLE factor_stock_daily_indicator AS
SELECT 
    a.ts_code, 
    a.trade_date, 
    a.total_mv, 
    a.turnover_rate, 
    a.volume_ratio, 
    a.pb, 
    a.pe_ttm, 
    a.ps_ttm, 
    a.dv_ttm, 
    b.pcf
FROM stock_daily_indicator a
LEFT JOIN stock_daily_pcf b
    ON a.ts_code = b.ts_code AND a.trade_date = b.trade_date
WHERE a.ts_code IN (SELECT ts_code FROM ts_code);


















