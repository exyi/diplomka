import pathlib
import os, sys, io, re, gzip, math, json, functools, itertools, numpy as np, typing as ty, argparse, multiprocessing.pool

import polars as pl
import pdb_utils
import para_utils
import pair_finding
import apply_filter
import pairs
import pair_defs
from simulate_fr3d_output import open_file_maybe, write_file as fr3d_write

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def _open_binary(fmt: str):
    if fmt.endswith('.gz') or fmt.endswith('.zst'):
        return True
    if fmt == 'parquet' or fmt == 'arrow':
        return 
    return False


def format_output(f: ty.Any, df: pl.DataFrame, format: str, header: bool, parameter_columns: ty.List[str] = []):
    if format.endswith('.gz'):
        import gzip
        with gzip.open(f, 'wt', compresslevel=9) as fGz:
            return format_output(fGz, df, format[:-3], header)

    if format.endswith('.zst'):
        import zstandard
        with zstandard.open(f, 'wt') as fZst:
            return format_output(fZst, df, format[:-4], header)

    nt_id_columns = [ 'chain', 'nr', 'res', 'alt', 'ins', 'symmetry_operation' ]
    id_columns = [ 'pdbid', 'model', 'family' ]
    class_col = (pl.col('family') + '-' + pl.col("res1").str.to_uppercase().replace(pair_defs.resname_map) + '-' + pl.col("res2").str.to_uppercase().replace(pair_defs.resname_map)).alias('class')
    parameter_columns = [ *parameter_columns, *[ c for c in df.columns if c.startswith('hb_') ] ]
    table_columns = [
        *id_columns,
        class_col,
        *(["accepted"] if "accepted" in df.columns else []),
        *[x + '1' for x in nt_id_columns],
        *[x + '2' for x in nt_id_columns],
        *parameter_columns
    ]
    json_columns = [
        *id_columns,
        *(["accepted"] if "accepted" in df.columns else []),
        pl.struct(pl.col(c + '1').alias(c) for c in nt_id_columns).alias('nt1'),
        pl.struct(pl.col(c + '2').alias(c) for c in nt_id_columns).alias('nt2'),
        pl.struct(pl.col(c) for c in parameter_columns).alias('params')
    ]
    jsoner = json.encoder.JSONEncoder(ensure_ascii=False, indent=None, separators=(', ', ': '))

    def json_lines(df: pl.DataFrame, include_pdbid: bool = True, include_params: bool = True):
        df = df.select(json_columns)
        if not include_pdbid:
            df = df.drop('pdbid', strict=False)
        if not include_params:
            df = df.drop('params', strict=False)
        def dict_drop_nulls(d):
            if not isinstance(d, dict):
                return d
            return { k: dict_drop_nulls(v) for k, v in d.items() if v is not None }
        for row in df.iter_rows(named=True):
            yield dict_drop_nulls(row)

    if format == 'csv':
        df.select(table_columns).write_csv(f, include_header=header)
    elif format == 'parquet':
        df.select(table_columns).write_parquet(f) # type:ignore
    elif format == 'arrow':
        df.select(table_columns).write_ipc(f) # type:ignore
    elif format == 'json':
        with open_file_maybe(f, 'wt') as io:
            io.write('{\n')
            for pdbid, dfg in df.group_by('pdbid'):
                io.write(f'  {jsoner.encode(pdbid)}: [\n')
                for i, row in enumerate(json_lines(dfg, include_pdbid=False)):
                    if i > 0:
                        io.write(',\n')
                    io.write('    ')
                    io.write(jsoner.encode(row))
                io.write('\n  ]')
            io.write('\n}\n')
    elif format == 'jsonl' or format == 'jsonnd':
        with open_file_maybe(f, 'wt') as io:
            for row in json_lines(df):
                io.write(jsoner.encode(row))
                io.write('\n')
    elif format == 'fr3d':
        fr3d_write(f, None, df, detailed=True, only_once=False, comparison_column=False, additional_columns=[])
    else:
        raise ValueError(f'Unknown output format {format}')
    
basic_parameters = [
    pairs.StandardMetrics.Coplanarity,
    # pairs.StandardMetrics.Isostericity,
    pairs.StandardMetrics.YawPitchRoll,
    pairs.StandardMetrics.YawPitchRoll2,
]

def analyze_structure(pdbid: str, min_atom_distance: float, boundaries: pl.DataFrame, best_fit: ty.Literal["none", "single-pair", "greedy-edges", "graph-edges"], output_all: bool) -> tuple[str, pl.DataFrame] | None:
    try:
        structure, sym_data, pdbid = pair_finding.load_structure(pdbid)
    except Exception as e:
        eprint(f"Error loading {pdbid}: {e}")
        return None

    df = pair_finding.find_contacts(structure, sym_data, 4.2)
    assert df is not None

    df = pairs.override_pair_family(df, pairs.all_familites)

    _, stat_columns, _ = pairs.make_stats_columns(pdbid, df, False, 4, basic_parameters, structure=structure)
    df = df.with_columns(stat_columns)

    df = pairs.postfilter_hb(df, 4.2)
    df = pairs.postfilter_shift(df, 2.2)
    df = pairs.remove_duplicate_pairs(df) # TODO: dedupe after main filter or before?
    df = apply_filter.apply_filter(df.lazy(), boundaries, False, "accepted" if output_all else None, best_fit)\
        .sort([pl.col("pdbid").str.to_lowercase(), "model", "chain1", "nr1", "alt1", "ins1", "chain2", "nr2", "alt2", "ins2", "symmetry_operation1", "symmetry_operation2", "family"])\
        .collect()
    return pdbid, df

def main_structure(structure: str, boundaries: pl.DataFrame, out: str | None, format: str, best_fit: ty.Literal["none", "single-pair", "greedy-edges", "graph-edges"], output_all: bool):
    """Loads the structure file, performs assignment and writes results"""
    x = analyze_structure(structure, 4.2, boundaries, best_fit, output_all)
    if x is None:
        return
    pdbid, df = x

    for fmt in format.split(','):
        file = os.path.join(out or '.', f'{pdbid}.{fmt}')
        format_output(file, df, fmt, True, [ c for p in basic_parameters for c in p.columns ])

def main2(pool: multiprocessing.pool.Pool | para_utils.MockPool, threads: int, args):
    inputs: list[str] = args.inputs
    out: str = args.out
    boundaries = pl.read_csv(args.boundaries, infer_schema_length=10000)

    os.makedirs(out, exist_ok=True)

    results = []
    for i, structure in enumerate(inputs):
        if args.verbose:
            eprint(f'Processing {structure} ({i+1}/{len(inputs)})')
        results.append(pool.apply_async(main_structure, (structure, boundaries, out, args.format, args.best_fit, args.output_all)))

    for r in results:
        r.get()

def main(args):
    pdb_utils.set_pdb_cache_dirs(args.pdbcache)

    threads = min(len(args.inputs), args.threads)
    if threads > 1:
        if args.verbose:
            eprint(f'Using {threads} worker processes')
        with multiprocessing.pool.ThreadPool(threads) as pool:
            main2(pool, threads, args)
    else:
        main2(para_utils.MockPool(), 1, args)

def parser(parser = None):
    parser = parser or argparse.ArgumentParser(description="""
        Performs basepair assignment for a given mmCIF structure or PDB ID.
        """)
    parser.add_argument('inputs', nargs='+', type=str, help='Input mmCIF/PDBx file or PDB ID to analyze')
    parser.add_argument('--out', type=str, help='Output directory for results.', required=True)
    parser.add_argument('-f', '--format', type=str, default='csv,parquet,fr3d,json', help='Output format(s) to generate. By default, all are selected. Supported formats are csv, parquet, fr3d, json, ndjson. One file per input structure will be created.')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--min-atom-distance', type=float, default=4.0, help='Distance threshold between any residue atoms (in Ångströms)')
    parser.add_argument('--boundaries', type=str, default='https://docs.google.com/spreadsheets/d/e/2PACX-1vQpIjMym1SejcSksbVfnV5WM89jYiR9PcmRcxiJd_0CihxZwVPN5vV-eH-w-dKS_ifCxcYNJqVc6HfG/pub?gid=245758142&single=true&output=csv', help='URL/filepath to the boundaries CSV file')
    parser.add_argument("--pdbcache", nargs="+", help="Directories to search for PDB files in order to avoid re-downloading. Last directory will be written to, if the structure is not found and has to be downloaded from RCSB. Also can be specified as PDB_CACHE_DIR env variable.")
    parser.add_argument("--best-fit", default='single-pair', type=str, help="Only the best fitting family for each basepair is kept", choices=["none", "single-pair", "greedy-edges", "graph-edges"])
    parser.add_argument("--output-all", action='store_true', help="Output all pairs of nucleotides which are close enough. This can be useful for debugging the parameter boundaries to see why a specific pair was *not* assigned. Basepairs which would be assigned as such will have assigned=true.")
    parser.add_argument('--threads', type=para_utils.parse_thread_count, default=1, help='Number of threads, 0 for all, 50%% for half, -1 to leave one free, ... Parallelism only affects processing of multiple PDB files.')
    return parser

if __name__ == '__main__':
    main(parser().parse_args())
