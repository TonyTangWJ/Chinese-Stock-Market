create table stock_daily_pcf (
    ts_code varchar(20) not null,
    trade_date varchar(20) not null,
    pcf decimal(20,4),
    primary key (ts_code, trade_date)
);