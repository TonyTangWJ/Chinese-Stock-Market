create table a_final_factor_daily(
    ts_code VARCHAR(20) NOT NULL,
    trade_date INT(10) NOT NULL,
    date INT(10) NOT NULL,
    total_mv DECIMAL(20,4),
    turnover_rate DECIMAL(20,4),
    volume_ratio DECIMAL(20,2),
    pb DECIMAL(20,4),
    pe_ttm DECIMAL(20,4),
    ps_ttm DECIMAL(20,4),
    dv_ttm DECIMAL(20,4),
    pcf DECIMAL(20,4),
    PRIMARY KEY (ts_code, trade_date, date)
);