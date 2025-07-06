-- operatin
-- caturn, fa_apnpturn, fa_arnrturn, fa_cashturnratio
CREATE TABLE factor_own_operatin AS
WITH operatin AS (
    SELECT 
        ts_code,
        end_date,
        CASE 
            WHEN EXTRACT(MONTH FROM end_date) = 3 THEN revenue
            ELSE revenue - LAG(revenue) OVER (PARTITION BY ts_code ORDER BY end_date)
        END AS q_revenue,
        CASE 
            WHEN EXTRACT(MONTH FROM end_date) = 3 THEN oper_cost
            ELSE oper_cost - LAG(oper_cost) OVER (PARTITION BY ts_code ORDER BY end_date)
        END AS q_COGS
    FROM 
        repo_income_statement
    where ts_code IN (SELECT ts_code FROM view_ts_code)
)
SELECT 
    a.ts_code,
    a.f_ann_date,
    a.end_date,
    q.q_revenue / NULLIF(((b.total_cur_assets + 
        COALESCE(LAG(b.total_cur_assets) OVER (PARTITION BY a.ts_code ORDER BY a.end_date), 
        b.total_cur_assets)) / 2.0), 0) AS caturn,
    q.q_revenue / NULLIF(((b.accounts_receiv + 
        COALESCE(LAG(b.accounts_receiv) OVER (PARTITION BY a.ts_code ORDER BY a.end_date), 
        b.accounts_receiv)) / 2.0), 0) AS fa_arnrturn,
    q.q_revenue / NULLIF(((b.money_cap + 
        COALESCE(LAG(b.money_cap) OVER (PARTITION BY a.ts_code ORDER BY a.end_date), 
        b.money_cap)) / 2.0), 0) AS fa_cashturnratio,
    q.q_COGS / NULLIF(((b.acct_payable + 
        COALESCE(LAG(b.acct_payable) OVER (PARTITION BY a.ts_code ORDER BY a.end_date), 
        b.acct_payable)) / 2.0), 0) AS fa_apnpturn
FROM 
    repo_income_statement a
JOIN 
    operatin q ON a.ts_code = q.ts_code AND a.end_date = q.end_date
JOIN 
    repo_balance_sheet b ON a.ts_code = b.ts_code AND a.end_date = b.end_date
ORDER BY
    a.ts_code, a.end_date
;



