
CREATE TABLE label_monthly (
    ts_code VARCHAR(20) NOT NULL,
    trade_date INT(10) NOT NULL,
    highest_return DECIMAL(10,4),
    lowest_return DECIMAL(10,4),
    close_return DECIMAL(10,4),
    PRIMARY KEY (ts_code, trade_date)
);






