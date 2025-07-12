-- growth
-- yoy_assets,yoydebt,pchequitypchgm_pchsale,chsaletomv

create table factor_own_growth as
SELECT
    a.ts_code,
    a.f_ann_date,
    a.end_date,
    (a.total_assets - LAG(a.total_assets, 4) OVER (PARTITION BY a.ts_code ORDER BY a.end_date)) 
        / NULLIF(LAG(a.total_assets, 4) OVER (PARTITION BY a.ts_code ORDER BY a.end_date), 0) * 100 AS yoy_assets,
    (a.total_liab - LAG(a.total_liab, 4) OVER (PARTITION BY a.ts_code ORDER BY a.end_date)) 
        / NULLIF(LAG(a.total_liab, 4) OVER (PARTITION BY a.ts_code ORDER BY a.end_date), 0) * 100 AS yoydebt,
    (a.total_hldr_eqy_exc_min_int - LAG(a.total_hldr_eqy_exc_min_int) OVER (PARTITION BY a.ts_code ORDER BY a.end_date))
        / NULLIF(LAG(a.total_hldr_eqy_exc_min_int) OVER (PARTITION BY a.ts_code ORDER BY a.end_date), 0) * 100 AS pchequity,
    (b.revenue - LAG(b.revenue, 4) OVER (PARTITION BY b.ts_code ORDER BY b.end_date)) 
        / NULLIF(LAG(b.revenue, 4) OVER (PARTITION BY b.ts_code ORDER BY b.end_date), 0) * 100 -
    (b.operate_profit - LAG(b.operate_profit, 4) OVER (PARTITION BY b.ts_code ORDER BY b.end_date)) 
        / NULLIF(LAG(b.operate_profit, 4) OVER (PARTITION BY b.ts_code ORDER BY b.end_date), 0) * 100 AS pchgm_pchsale
FROM
    repo_balance_sheet a
JOIN
    repo_income_statement b ON a.ts_code = b.ts_code AND a.end_date = b.end_date
WHERE
    a.ts_code IN (SELECT ts_code FROM ts_code)
ORDER BY
    a.ts_code, a.end_date;




