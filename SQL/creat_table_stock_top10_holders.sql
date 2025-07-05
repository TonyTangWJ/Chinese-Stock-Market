CREATE TABLE stock_top10_holders (
    ts_code VARCHAR(20) NOT NULL,
    ann_date VARCHAR(20) NOT NULL,
    end_date VARCHAR(20) NOT NULL,
    holder_name VARCHAR(100) NOT NULL,
    hold_amount DECIMAL(20,4),
    hold_ratio DECIMAL(10,4),
    hold_float_ratio DECIMAL(10,4),
    hold_change DECIMAL(20,4),
    holder_type VARCHAR(50),
    PRIMARY KEY (ts_code, ann_date, end_date, holder_name, hold_amount),
    foreign key (ts_code) REFERENCES stock_list(ts_code)
);