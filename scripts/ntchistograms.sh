#!/usr/bin/env bash

series="$(seq --separator=, 0 0.01 1)"
file="data/all-csvs-bp.zst.parquet"

function hist_q {
    echo "quantile_cont($1, quantile_list()) as $1_hist"
}
duckdb <<END
create macro quantile_list() as [$series];
copy (select NtC, count(*) as count,
    $(hist_q d1),
    $(hist_q d2),
    $(hist_q ch1),
    $(hist_q ch2),
    $(hist_q rmsd)
from '$file'
group by NtC
order by NtC)
TO 'ntc_quantiles.parquet' (FORMAT PARQUET);
END
    
