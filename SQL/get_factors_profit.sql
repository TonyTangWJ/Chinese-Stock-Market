-- profit
-- pchsaletoasset, pchtax
create table factor_own_profit as
WITH profit AS (
    SELECT 
        ts_code,
        f_ann_date,
        end_date,
        CASE 
            WHEN EXTRACT(MONTH FROM end_date) = 3 THEN revenue
            ELSE revenue - LAG(revenue) OVER (PARTITION BY ts_code ORDER BY end_date)
        END AS q_revenue,
        income_tax
    FROM 
        repo_income_statement
    WHERE ts_code IN (SELECT ts_code FROM view_ts_code)
)
SELECT
    a.ts_code,
    a.f_ann_date,
    a.end_date,
    (a.q_revenue - LAG(a.q_revenue) OVER (PARTITION BY a.ts_code ORDER BY a.end_date)) / NULLIF(b.total_assets, 0) AS pchsaletoasset,
    (COALESCE(a.income_tax, 0) - COALESCE(LAG(a.income_tax) OVER (PARTITION BY a.ts_code ORDER BY a.end_date), 0)) AS pchtax
FROM
    profit a
JOIN
    repo_balance_sheet b ON a.ts_code = b.ts_code AND a.end_date = b.end_date
WHERE
    a.ts_code IN (SELECT ts_code FROM ts_code)
ORDER BY
    a.ts_code, a.end_date;

