-- get selected ts_code

create view view_ts_code as 
    select DISTINCT ts_code
    from stock_list
    where market = 'MainBoard'
    and ts_code in (
        select distinct ts_code
        from repo_balance_sheet
        where comp_type = '1'
    )