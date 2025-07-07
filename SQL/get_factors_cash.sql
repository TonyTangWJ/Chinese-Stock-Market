-- cash
-- cftoassets,fcftocf,icftocf,netprofitcashcover,ocftocf

create table factor_own_cash as
WITH cash AS (
    SELECT 
        ts_code,
        end_date,
        CASE 
            WHEN EXTRACT(MONTH FROM end_date) = 3 THEN n_income
            ELSE n_income - LAG(n_income) OVER (PARTITION BY ts_code ORDER BY end_date)
        END AS q_n_income,
        COALESCE(n_cashflow_act,0)+COALESCE(n_cashflow_inv_act,0)+COALESCE(n_cash_flows_fnc_act,0) AS q_n_cash
    FROM 
        repo_income_statement
    JOIN
        repo_cashflow_statement USING (ts_code, end_date)
    where ts_code IN (SELECT ts_code FROM ts_code)
)
select
    a.ts_code,
    a.f_ann_date,
    a.end_date,
    b.money_cap / NULLIF(((b.total_assets + 
        COALESCE(LAG(b.total_assets) OVER (PARTITION BY a.ts_code ORDER BY a.end_date), 
        b.total_assets)) / 2.0),0) AS cftoassets,
    COALESCE(a.n_cashflow_act,0) / NULLIF(c.n_income,0) AS netprofitcashcover,
    COALESCE(n_cashflow_act,0) / NULLIF(d.q_n_cash,0) AS ocftocf,
    COALESCE(n_cashflow_inv_act,0) / NULLIF(d.q_n_cash,0) AS icftocf,
    COALESCE(n_cash_flows_fnc_act,0) / NULLIF(d.q_n_cash,0) AS fcftocf
FROM
    repo_cashflow_statement a
JOIN
    repo_balance_sheet b ON a.ts_code = b.ts_code AND a.end_date = b.end_date
JOIN    
    repo_income_statement c ON a.ts_code = c.ts_code AND a.end_date = c.end_date
JOIN
    cash d ON a.ts_code = d.ts_code AND a.end_date = d.end_date
ORDER BY
    a.ts_code, a.end_date
;



