-- Combined SQL: View with filtered data for Mainboard stocks with balance sheet data

CREATE OR REPLACE VIEW target_data AS
SELECT 
    ts_code,
    trade_date,
    ROUND(
        GREATEST(
            (
                (high - LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date)) 
                / LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date)
            ) * 100,
            0
        ), 
        2
    ) AS highest_return,
    
    ROUND(
        LEAST(
            (
                (low - LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date)) 
                / LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date)
            ) * 100,
            0
        ), 
        2
    ) AS lowest_return,
    
    ROUND(
        (
            (close - LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date)) 
            / LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date)
        ) * 100,
        2
    ) AS close_return
    
FROM stock_monthly_trading_data
WHERE ts_code IN (
    SELECT DISTINCT ts_code
    FROM stock_list 
    WHERE market = 'Mainboard' 
    AND ts_code IN (
        SELECT DISTINCT ts_code 
        FROM repo_balance_sheet 
        WHERE comp_type = 1
    )
);








