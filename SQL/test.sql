-- truncate table stock_daily_pcf;
select * from stock_daily_indicator
where ts_code in (select ts_code from ts_code)









