
import polars as pl
csv_schema = [
    ('pdbid', pl.Utf8),
    ('model', pl.Int64),
    ('chain1', pl.Utf8),
    ('res1', pl.Utf8),
    ('nr1', pl.Int64),
    ('ins1', pl.Utf8),
    ('atom1', pl.Utf8),
    ('alt1', pl.Utf8),
    ('chain2', pl.Utf8),
    ('res2', pl.Utf8),
    ('nr2', pl.Int64),
    ('ins2', pl.Utf8),
    ('atom2', pl.Utf8),
    ('alt2', pl.Utf8),
    ('image', pl.Int64),
    ('dist', pl.Float64),
    ('rmsd', pl.Float64),
    ('tx', pl.Float64),
    ('ty', pl.Float64),
    ('tz', pl.Float64),
    ('r11', pl.Float64),
    ('r12', pl.Float64),
    ('r13', pl.Float64),
    ('r21', pl.Float64),
    ('r22', pl.Float64),
    ('r23', pl.Float64),
    ('r31', pl.Float64),
    ('r32', pl.Float64),
    ('r33', pl.Float64),
    ('t1MMB', pl.Float64),
    ('t2MMB', pl.Float64),
    ('t3MMB', pl.Float64),
    ('theta', pl.Float64),
    ('axMMB', pl.Float64),
    ('ayMMB', pl.Float64),
    ('azMMB', pl.Float64),
    # ('polar_dist', pl.Float64),
    # ('polar_atom1', pl.Utf8),
    # ('polar_atom2', pl.Utf8),
    # ('tripletx', pl.Utf8),
    # ('triplety', pl.Utf8),
    # ('tripletz', pl.Utf8),
    # ('pdbsymstr', pl.Utf8),
    # ('gemmisymstr', pl.Utf8),
    # ('gemmi_shortest', pl.Float64)
]

def scan_pair_csvs(files: list[str], header=None):
    if header is None:
        with open(files[0], "rt") as f:
            header_maybe = f.readline()
            header = header_maybe.startswith("pdbid,")

    if header:
        dtypes = dict(csv_schema)
    else:
        dtypes = { f'column_{1+i}': t for i, (n, t) in enumerate(csv_schema) }
    df = pl.concat([ pl.scan_csv(f, has_header=False, dtypes=dtypes) for f in files ])
    if not header:
        df = df.select([ pl.col(f'column_{1+i}').alias(n) for i, (n, t) in enumerate(csv_schema) ])
    # df = df.filter(pl.col("tripletx").eq("x") & pl.col("triplety").eq("y") & pl.col("tripletz").eq("z") & pl.col("pdbsymstr").eq("1_555"))
    return df
