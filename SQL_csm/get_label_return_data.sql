CREATE ALGORITHM=UNDEFINED 
DEFINER=`root`@`localhost` 
SQL SECURITY DEFINER 
VIEW `target_data` AS 
SELECT 
    `stock_monthly_trading_data`.`ts_code` AS `ts_code`,
    `stock_monthly_trading_data`.`trade_date` AS `trade_date`,
    
    -- 计算最高收益率，取最大值并四舍五入到两位小数
    ROUND(
        GREATEST(
            (
                (
                    (`stock_monthly_trading_data`.`high` - 
                     LAG(`stock_monthly_trading_data`.`close`) 
                     OVER (PARTITION BY `stock_monthly_trading_data`.`ts_code` 
                           ORDER BY `stock_monthly_trading_data`.`trade_date`)
                    ) 
                    / 
                    LAG(`stock_monthly_trading_data`.`close`) 
                    OVER (PARTITION BY `stock_monthly_trading_data`.`ts_code` 
                          ORDER BY `stock_monthly_trading_data`.`trade_date`)
                ) * 100
            ), 
            0
        ), 
        2
    ) AS `highest_return`,
    
    -- 计算最低收益率，取最小值并四舍五入到两位小数
    ROUND(
        LEAST(
            (
                (
                    (`stock_monthly_trading_data`.`low` - 
                     LAG(`stock_monthly_trading_data`.`close`) 
                     OVER (PARTITION BY `stock_monthly_trading_data`.`ts_code` 
                           ORDER BY `stock_monthly_trading_data`.`trade_date`)
                    ) 
                    / 
                    LAG(`stock_monthly_trading_data`.`close`) 
                    OVER (PARTITION BY `stock_monthly_trading_data`.`ts_code` 
                          ORDER BY `stock_monthly_trading_data`.`trade_date`)
                ) * 100
            ), 
            0
        ), 
        2
    ) AS `lowest_return`,
    
    -- 计算收盘收益率并四舍五入到两位小数
    ROUND(
        (
            (
                (`stock_monthly_trading_data`.`close` - 
                 LAG(`stock_monthly_trading_data`.`close`) 
                 OVER (PARTITION BY `stock_monthly_trading_data`.`ts_code` 
                       ORDER BY `stock_monthly_trading_data`.`trade_date`)
                ) 
                / 
                LAG(`stock_monthly_trading_data`.`close`) 
                OVER (PARTITION BY `stock_monthly_trading_data`.`ts_code` 
                      ORDER BY `stock_monthly_trading_data`.`trade_date`)
            ) * 100
        ), 
        2
    ) AS `close_return`
FROM 
    `stock_monthly_trading_data`
WHERE 
    `stock_monthly_trading_data`.`ts_code` IN (
        SELECT DISTINCT `stock_list`.`ts_code`
        FROM `stock_list`
        WHERE 
            `stock_list`.`market` = 'Mainboard' 
            AND `stock_list`.`ts_code` IN (
                SELECT DISTINCT `repo_balance_sheet`.`ts_code`
                FROM `repo_balance_sheet`
                WHERE `repo_balance_sheet`.`comp_type` = 1
            )
    );