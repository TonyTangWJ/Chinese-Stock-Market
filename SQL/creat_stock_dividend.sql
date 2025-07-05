CREATE TABLE stock_dividend (
    ts_code VARCHAR(20) NOT NULL,
    ann_date VARCHAR(20) NOT NULL,
    end_date VARCHAR(20) NOT NULL,
    div_proc VARCHAR(20),
    stk_div DECIMAL(20,4),
    stk_bo_rate DECIMAL(20,4),
    stk_co_rate DECIMAL(20,4),
    cash_div DECIMAL(20,4),
    cash_div_tax DECIMAL(20,4),
    base_share DECIMAL(20,4),
    PRIMARY KEY (ts_code, ann_date, end_date),
    foreign key (ts_code) REFERENCES stock_list(ts_code)
);
