create table factor_own_structure as
SELECT
    ts_code,
    end_date,
    total_ncl / NULLIF(total_hldr_eqy_exc_min_int, 0) AS longdebttoequity
FROM
    repo_balance_sheet
WHERE
    ts_code IN (SELECT ts_code FROM ts_code)
ORDER BY
    ts_code, end_date;