-- creates a target data used for machine learning.

CREATE TABLE a_final_label_monthly(
    ts_code VARCHAR(20) not null,
    trade_date int(10) not null,
    highest_return decimal(20,4),
    lowest_return decimal(20,4),
    close_return decimal(20,4),
    primary key (ts_code, trade_date)
)


