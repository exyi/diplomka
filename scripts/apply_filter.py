#!/usr/bin/env python3

from multiprocessing.pool import Pool
from types import NoneType
import typing as ty
import os, sys, io, re, gzip, math, json, functools, itertools, numpy as np
import polars as pl
from dataclasses import dataclass
import dataclasses
from pair_csv_parse import scan_pair_csvs
import pair_defs as pair_defs


def save_output(args, df: pl.LazyFrame):
    df = df.drop(c for c in df.columns if re.match("^_tmp.*$", c))
    if args.output.endswith(".parquet"):
        df.sink_parquet(args.output)
    elif args.output != "/dev/null":
        df.sink_csv(args.output if args.output.endswith(".csv") else args.output + ".csv")
        df.sink_parquet(args.output + ".parquet")
    return df

def main(args):
    boundaries = pl.read_csv(args.boundaries, infer_schema_length=10000)
    df = scan_pair_csvs(args.inputs)

    df = apply_filter(df, boundaries, args.null_is_fine, args.accepted_column, args.best_fit)

    df = df.sort("pdbid", "model", "chain1", "nr1", "alt1", "ins1", "chain2", "nr2", "alt2", "ins2", "symmetry_operation1", "symmetry_operation2", "family")

    save_output(args, df)

def apply_filter(df: pl.LazyFrame, boundaries: pl.DataFrame, null_is_fine: bool, accepted_column: str | None, best_fit: ty.Literal["none", "single-pair", "greedy-edges", "graph-edges"]) -> pl.LazyFrame:
    """
    Filters `df` according to the limits in `boundaries`.

    Additional arguments:
    - `null_is_fine` - if a column is NULL, treat is as "in range". When null_is_fine=False, all rows will null columns are rejected.
    - `accepted_column` - instead of removing the non-matching rows, create a new boolean column with the specified name and set it to True, when the pair matches the criteria in `boundaries`
    - best_fit - See --best-fit help
    """
    columns = df.collect_schema().names()
    if "hb_0_length" not in columns or "C1_C1_yaw1" not in columns:
        raise ValueError("The input data is missing the necessary columns")

    df = df.with_columns(
        _tmp_bases=pl.col("res1").replace(pair_defs.resname_map).str.to_uppercase() + "-" + pl.col("res2").replace(pair_defs.resname_map).str.to_uppercase(),
        _tmp_min_bond_length = pl.min_horizontal(pl.col("^hb_\\d+_length$")),
    )
    columns = df.collect_schema().names()

    colname_mapping = {
        "yaw1": "C1_C1_yaw1",
        "yaw2": "C1_C1_yaw2",
        "pitch1": "C1_C1_pitch1",
        "pitch2": "C1_C1_pitch2",
        "roll1": "C1_C1_roll1",
        "roll2": "C1_C1_roll2",
        "min_bond_length": "_tmp_min_bond_length",
    }

    checked_columns = set(c[:-4] for c in boundaries.columns if c.endswith("_min") or c.endswith("_max"))
    checked_columns = set(colname_mapping.get(c, c) for c in checked_columns)
    if len(missing_columns := checked_columns - set(columns)) > 0:
        print(f"WARNING: The data is missing the following columns with boundaries specified: {[*missing_columns]}")
        checked_columns.difference_update(missing_columns)

    # Create a Polars logical expression for each basepair family which checks that all parameters are in range
    prec_tolerance = 0.01
    conditions: dict[pair_defs.PairType, pl.Expr] = dict()
    for b in boundaries.iter_rows(named=True):
        pair_type = pair_defs.PairType.from_tuple((b["family"], b["bases"])).normalize_capitalization()
        # hbond = pair_defs.get_hbonds(pair_type)
        # hbond_default_lengths = [ pair_defs.is_ch_bond(pair_type, hb) or pair_defs.is_bond_to_sugar(pair_type, hb) for hb in hbond ]

        row_conditions = []
        for c in checked_columns:
            min, max = (b.get(c + "_min", None), b.get(c + "_max", None))
            if not isinstance(min, (int, float, NoneType)):
                raise ValueError(f"Invalid value for {c}_min ({repr(min)} of type {type(min)})")
            if not isinstance(max, (int, float, NoneType)):
                raise ValueError(f"Invalid value for {c}_max ({repr(max)} of type {type(max)})")
            if min is None and max is None:
                continue
            if min is None:
                row_conditions.append(pl.col(c) <= max + prec_tolerance)
            elif max is None:
                row_conditions.append(pl.col(c) >= min - prec_tolerance)
            elif min > max:
                # inverted range - for angular -180..180 columns
                row_conditions.append(pl.col(c).is_between(max - prec_tolerance, min + prec_tolerance).not_())
            else:
                row_conditions.append(pl.col(c).is_between(min - prec_tolerance, max + prec_tolerance))

            if null_is_fine:
                row_conditions[-1] = row_conditions[-1] | pl.col(c).is_null()

        if len(row_conditions) == 0:
            print(f"WARNING: No boundaries specified for {pair_type}")
            continue

        conditions[pair_type] = pl.all_horizontal(row_conditions)


    # combine all pair_type conditions into a single conditional expression:
    # if (_tmp_bases == 'A-A' and family == 'cWW')
    #    then ...
    # elseif (_tmp_bases == 'A-C' and family == 'cWW')
    #    then ...
    # ...
    monster_condition = pl.when(pl.lit(False)).then(False)
    for pt, c in conditions.items():
        monster_condition = monster_condition.when(
            pl.all_horizontal(
                pl.col("_tmp_bases") == pt.bases_str.upper(),
                pl.col("family").str.to_uppercase() == pt.type.upper(),
            )).then(c)
    monster_condition = monster_condition.otherwise(False)

    if accepted_column is not None:
        # add `accepted` column with the result of the condition
        df = df.with_columns(monster_condition.fill_null(False).alias(accepted_column))
    else:
        df = df.filter(monster_condition)

    if best_fit is not None and best_fit != "none":
        df = best_fitting_nucleotides(df, best_fit, accepted_column)

    return df

conflict_resolution_score = ((pl.sum_horizontal(pl.col("^hb_\\d+_length$") - 4).fill_null(0)) + pl.col("coplanarity_shift1") + pl.col("coplanarity_shift2")).alias("conflict_resolution_score")

def edge_df(df: pl.DataFrame, i: ty.Literal[1, 2]) -> pl.DataFrame:
    return df.select(
        pl.col("family").str.slice(i, 1), # edge
        pl.col("pdbid"),
        pl.col(f"symmetry_operation{i}"),
        pl.col("model"),
        pl.col(f"chain{i}"),
        pl.col(f"nr{i}"),
        pl.col(f"ins{i}"),
        pl.col(f"alt{i}"),
    )


def best_fitting_edges_optmatching(df: pl.DataFrame) -> pl.DataFrame:
    try:
        import networkx as nx
        bitmap = np.zeros(len(df), dtype=bool)
        scores = df.select(conflict_resolution_score)[:, 0]

        graph = nx.Graph()
        for e1, e2, score in zip(edge_df(df, 1).iter_rows(), edge_df(df, 2).iter_rows(), scores):
            weight = 2 - score
            weight = max(0.01, weight)
            # print("Adding edge", e1, e2, weight)
            graph.add_edge(e1, e2, weight=weight)

        # find pairs connected to alternative residues and contract said alternatives into one
        for res1 in list(graph.nodes):
            neighbors = list(graph.neighbors(res1))
            if len(neighbors) <= 1:
                continue

            no_alt_neighbors = set((edge, pdbid, symop, model, chain, nr, ins) for edge, pdbid, symop, model, chain, nr, ins, _alt in neighbors)
            if len(neighbors) == len(no_alt_neighbors):
                continue

            groups = dict()
            for edge, pdbid, symop, model, chain, nr, ins, alt in neighbors:
                groups.setdefault((pdbid, symop, model, chain, nr, ins), []).append((edge, pdbid, symop, model, chain, nr, ins, alt))

            for g in groups.values():
                if len(g) > 1:
                    print("Contracting", g)
                    for i in range(len(g) - 1):
                        nx.contracted_nodes(graph, g[i], g[i + 1], self_loops=False, copy=False)
                    altcodes = "merged" + "".join(gi[-1] for gi in g)
                    nx.relabel_nodes(graph, {g[-1]: (*g[-1][:-1], altcodes)}, copy=False)

        opt_edges: set[tuple[tuple, tuple]] = nx.max_weight_matching(graph)
        # print("Optimal edges:", opt_edges)
        opt_edges.update(list((b, a) for a, b in opt_edges))

        for i, (e1, e2) in enumerate(zip(edge_df(df, 1).iter_rows(), edge_df(df, 2).iter_rows())):
            if (e1, e2) in opt_edges:
                bitmap[i] = True

        print(f"Selected {bitmap.sum()} out of {len(df)} basepairs")
        return df.filter(pl.Series(bitmap, dtype=pl.Boolean))
    except Exception as e:
        print("Error in best_fitting_nucleotide_single_structure:", e)
        import traceback
        traceback.print_exc()
        raise

def best_fitting_edges_greedy(df: pl.DataFrame) -> pl.DataFrame:
    try:
        df = df.sort(conflict_resolution_score)
        bitmap = np.zeros(len(df), dtype=bool)
        dedup = set()

        for i, (e1, e2) in enumerate(zip(edge_df(df, 1).iter_rows(), edge_df(df, 2).iter_rows())):
            if e1 not in dedup and e2 not in dedup:
                bitmap[i] = True
                dedup.add(e1)
                dedup.add(e2)

        print(f"Selected {bitmap.sum()} out of {len(df)} basepairs")
        return df.filter(pl.Series(bitmap, dtype=pl.Boolean))
    except Exception as e:
        print("Error in best_fitting_nucleotide_single_structure:", e)
        import traceback
        traceback.print_exc()
        raise


def best_fitting_nucleotides(df: pl.LazyFrame, method: str, accepted_column: str|None) -> pl.LazyFrame:
    """
    Includes only the best fitting basepair class
    * for each nucleotide pair.
    * for each nucleotide edge (if method != 'single-pair')
    """
    if accepted_column is not None:
        orig_df = df
        df = df.filter(pl.col(accepted_column))
    df = df.sort(conflict_resolution_score)

    # first, select best matching assignment for each pair
    id_columns = ["family", pl.col("pdbid").str.to_lowercase(), "model", "chain1", "nr1", "alt1", "ins1", "chain2", "nr2", "alt2", "ins2", "symmetry_operation1", "symmetry_operation2"]
    gr = df.group_by(id_columns, maintain_order=True)
    df = gr.first()

    # second, select best matching partner for each nt/edge
    if method == "single-pair":
        pass
    elif method == "greedy-edges":
        df = df.group_by(["pdbid"]).map_groups(best_fitting_edges_optmatching, schema=None)
    elif method == "graph-edges":
        df = df.group_by(["pdbid"]).map_groups(best_fitting_edges_greedy, schema=None)

    df = df.collect().lazy()

    if accepted_column is not None:
        df = orig_df.drop(accepted_column, strict=False).join(df.with_columns(pl.lit(True).alias(accepted_column)),
            left_on=id_columns,
            right_on=id_columns,
            how="left",
            suffix="_right",
            join_nulls=True
        )
        assert f"{accepted_column}_right" not in df.collect_schema().names()
        df = df.drop("^.*_right$")
        df = df.with_columns(pl.col(accepted_column).fill_null(False))

    return df

def parser(parser = None):
    import argparse
    parser = parser or argparse.ArgumentParser(description="""
        Filters the input CSV/Parquet file according to the specified boundaries files (columns ending with _min and _max).
        """,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", "-o", required=True, help="Output CSV/Parquet file name.")
    parser.add_argument("--boundaries", type=str, required=False, default="https://docs.google.com/spreadsheets/d/e/2PACX-1vQpIjMym1SejcSksbVfnV5WM89jYiR9PcmRcxiJd_0CihxZwVPN5vV-eH-w-dKS_ifCxcYNJqVc6HfG/pub?gid=245758142&single=true&output=csv", help="Input file with boundaries. May be URL and the Google table is default.")
    parser.add_argument("--null-is-fine", default=False, action="store_true", help="Columns with NULL values are considered to be within boundaries. Rows with all NULL columns are still discarded")
    parser.add_argument("--best-fit", default=None, type=str, help="Only the best fitting family for each basepair is kept", choices=["none", "single-pair", "greedy-edges", "graph-edges"])
    parser.add_argument("--accepted-column", default=None, type=str, help="If specified, instead of filtering the file, add a boolean column indication whether the pair is accepted or not")
    return parser


if __name__ == "__main__":
    main(parser().parse_args())

