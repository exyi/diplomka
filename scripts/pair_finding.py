import multiprocessing.pool
from typing import Any, Callable, Generator, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple, TypeVar, Union
import Bio.PDB, Bio.PDB.Structure, Bio.PDB.Model, Bio.PDB.Residue, Bio.PDB.Atom, Bio.PDB.Chain, Bio.PDB.Entity
import os, sys, io, gzip, math, json, numpy as np, re, functools, itertools
import multiprocessing, para_utils
import pdb_utils
import polars as pl

import scipy.spatial.distance

# library for finding close residues

coord_dtype = np.float64

def _coord(grid_size: float, coord) -> tuple[int, int, int]:
    cs = np.floor(np.divide(coord, grid_size))
    assert cs.size == 3
    return tuple(int(c) for c in cs) # type: ignore

AtomGrid = dict[tuple[int, int, int], list[Bio.PDB.Atom.Atom]]
CoordGrid = dict[tuple[int, int, int], np.ndarray]

def get_atom_grid(size: float, atoms: Iterator[Bio.PDB.Atom.Atom]) -> AtomGrid:
    grid: AtomGrid = dict()
    def add_atom(a: Bio.PDB.Atom.Atom):
        coord = _coord(size, a.coord)
        if not (list := grid.get(coord, None)):
            grid[coord] = (list := [])
        list.append(a)
    for a in atoms:
        if isinstance(a, Bio.PDB.Atom.DisorderedAtom):
            for a2 in a.disordered_get_list():
                add_atom(a2)
        else:
            add_atom(a)
    return grid

def to_coord_grid(g: AtomGrid) -> CoordGrid:
    return {
        k: np.array([ a.coord for a in v ], dtype=coord_dtype)
        for k, v in g.items()
    }

_directions = [
    (int(i[0]-1), int(i[1]-1), int(i[2]-1))
    for i, _ in np.ndenumerate(np.zeros((3, 3, 3)))
]

_na_residues = { 'A', 'T', 'G', 'C', 'U', 'DA', 'DT', 'DU', 'DG', 'DC', '5MC' }

def _matches_wblist[T](whitelist: Iterable[T] | None, blacklist: Iterable[T] | None, x: T) -> bool:
    """
    Does x match the whitelist and blacklist? When whitelist is None, all values not in blacklist are allowed.
    """
    if blacklist is not None and x in blacklist:
        return False

    if whitelist is not None and x not in whitelist:
        return False
    return True

class StructureIndex:
    """
    Index for finding close atoms in a structure.

    The structure is divided into a grid of cells with the specified `distance` between them.
    Finding atoms within the `distance` then involves only looking into the 9 neighboring cells.
    """
    def __init__(self, distance: float, atoms: Iterator[Bio.PDB.Atom.Atom]) -> None:
        self.distance = distance
        self.sqdistance = distance ** 2
        self.grid = get_atom_grid(distance, atoms)
        self.coord_grid = to_coord_grid(self.grid)

    @staticmethod
    def from_model(
        distance: float,
        s: Bio.PDB.Model.Model,
        residue_whitelist: Optional[set[str]] = _na_residues,
        residue_blacklist: Optional[set[str]] = None):

        def enum_atoms():
            r: Bio.PDB.Residue.Residue
            for r in s.get_residues():
                if not _matches_wblist(residue_whitelist, residue_blacklist, r.resname):
                    continue
                for a in r.get_unpacked_list():
                    yield a

        return StructureIndex(distance, enum_atoms())

    def iter_neighbor_atoms(self, coord: np.ndarray) -> Iterator[Bio.PDB.Atom.Atom]:
        if len(coord.shape) == 1:
            coord = coord.reshape((1, 3))
        assert len(coord.shape) == 2 and coord.shape[1] == 3

        cell_coords = np.floor(coord / self.distance).astype(np.int32)
        cell_coords = np.unique(cell_coords, axis=0)
        cell_coords = np.unique(np.concatenate([ cell_coords + d for d in _directions ]), axis=0)
        sqdistance = self.sqdistance

        for cc in cell_coords:
            cc = tuple(cc)
            assert len(cc) == 3
            cell = self.coord_grid.get(cc)
            if cell is None:
                continue
            
            # print(np.min(scipy.spatial.distance.cdist(coord, cell, 'sqeuclidean'), axis=0))
            atom_bitmap = np.min(scipy.spatial.distance.cdist(coord, cell, 'sqeuclidean'), axis=0) < sqdistance
            if np.any(atom_bitmap):
                atoms = self.grid[cc]
                assert len(atoms) == cell.shape[0]
                for i in np.where(atom_bitmap)[0]:
                    yield atoms[i]

    def find_neighbor_residues(self, coord: np.ndarray) -> set[Bio.PDB.Residue.Residue]:
        result: set[Bio.PDB.Residue.Residue] = set()
        for a in self.iter_neighbor_atoms(coord):
            assert isinstance(a.parent, Bio.PDB.Residue.Residue), f"a.parent is not Residue: {a.full_id}"

            result.add(a.parent)

        return result
    
    def iter_contact_candidates(self,
        s: Bio.PDB.Model.Model,
        residue_whitelist: Optional[set[str]] = _na_residues,
        residue_blacklist: Optional[set[str]] = None, # { 'HOH' }
    ) -> Iterator[tuple[Bio.PDB.Residue.Residue, Bio.PDB.Residue.Residue]]:
        r: Bio.PDB.Residue.Residue
        for r in s.get_residues():
            if not _matches_wblist(residue_whitelist, residue_blacklist, r.resname):
                continue

            coord = np.array([ a.coord for a in r.get_unpacked_list() ], dtype=coord_dtype)
            # print(coord)
            for neighbor in self.find_neighbor_residues(coord):
                if r != neighbor:
                    yield (r, neighbor)

def _get_alts(r: Bio.PDB.Residue.Residue) -> list[str]:
    if not r.is_disordered():
        return [ '' ]
    datoms = (a for a in r.get_unpacked_list() if isinstance(a, Bio.PDB.Atom.DisorderedAtom))
    dnames = set(d for a in datoms for d in a.disordered_get_id_list())
    return list(sorted(dnames))


def _to_contact_df(index: StructureIndex, s: Bio.PDB.Model.Model, sym: Optional[pdb_utils.SymmetryOperation]) -> pl.DataFrame:
    """
    Finds contacts between the indexed structure and `s` in symmetry `sym`.

    Returns DataFrame with columns chain1, res1, nr1, ins1, alt1, chain2, res2, nr2, ins2, alt2, pdbid, model, symmetry_operation1, symmetry_operation2.
    """
    pdbid = s.parent.id
    if sym and sym.pdbname != '1_555':
        s = s.copy()
        s.transform(sym.rotation, sym.translation)

    result = []
    for r1, r2 in index.iter_contact_candidates(s):
        # we need the duplicates currently
        # if sym is None and r1.full_id > r2.full_id:
        #     continue
        for alt1 in _get_alts(r1):
            for alt2 in _get_alts(r2):
                result.append({
                    'chain1': r1.parent.id,
                    'res1': r1.resname,
                    'nr1': r1.id[1],
                    'ins1': r1.id[2],
                    'alt1': alt1,
                    'chain2': r2.parent.id,
                    'res2': r2.resname,
                    'nr2': r2.id[1],
                    'ins2': r2.id[2],
                    'alt2': alt2,
                })
    
    return pl.DataFrame(result, schema={
        'chain1': pl.Utf8,
        'res1': pl.Utf8,
        'nr1': pl.Int32,
        'ins1': pl.Utf8,
        'alt1': pl.Utf8,
        'chain2': pl.Utf8,
        'res2': pl.Utf8,
        'nr2': pl.Int32,
        'ins2': pl.Utf8,
        'alt2': pl.Utf8,
    }).with_columns(
        pdbid=pl.lit(pdbid.lower(), pl.Utf8),
        model=pl.lit(s.id + 1, pl.Int32),
        symmetry_operation1=pl.lit(None, pl.Utf8),
        symmetry_operation2=pl.lit(sym.pdbname if sym and sym.pdbname != '1_555' else None, pl.Utf8),
    )

def load_structure(pdb: str) -> tuple[Bio.PDB.Structure.Structure, pdb_utils.StructureData, str]:
    if "." in pdb:
        pdbid = re.search(r"(\w{4})[.](pdb|cif|mmcif)([.](gz|zstd?))?$", pdb, re.IGNORECASE).group(1)
        s = pdb_utils.load_pdb(pdb, pdbid.lower())
        sym = pdb_utils.load_sym_data(pdb, pdbid)
    else:
        s = pdb_utils.load_pdb(None, pdb)
        sym = pdb_utils.load_sym_data(None, pdb)

    pdbid: str = s.id
    pdbid = pdbid.lower()

    return s, sym, pdbid

def find_contacts(s: Bio.PDB.Structure.Structure, sym: pdb_utils.StructureData, distance: float) -> pl.DataFrame:
    """Finds contacting residues in the structure `s` within `distance` Ångströms."""
    result_dfs: list[pl.DataFrame] = []
    for model_i, m in enumerate(s.get_models()):
        ix = StructureIndex.from_model(distance, m)
        result_dfs.append(_to_contact_df(ix, m, None))
        for symop in sym.assembly:
            if symop.pdbname != '1_555':
                result_dfs.append(_to_contact_df(ix, m, symop))

    return pl.concat(result_dfs)


def main_1structure(pdb: str, output_directory: str | None, distance: float):
    try:
        s, sym, pdbid = load_structure(pdb)
    except Exception as e:
        print(f"Error loading {pdb}: {e}")
        return

    result_df = find_contacts(s, sym, distance)

    if output_directory is not None:
        result_df.write_parquet(os.path.join(output_directory, f"{pdbid}.parquet"))
        print(f"Processed {pdbid}: {len(result_df)} contacts ({len(result_df.filter(pl.col("symmetry_operation2").is_not_null()))} with symmetry operation)")
    else:
        return result_df
    

def _main(args, pool: Union[para_utils.MockPool, multiprocessing.pool.Pool]):
    if os.path.exists(args.output):
        if os.listdir(args.output):
            raise ValueError(f"Output directory {args.output} is not empty")
    else:
        os.makedirs(args.output)
    
    inputs = args.inputs
    r = [ pool.apply_async(main_1structure, args=[pdb, args.output, args.distance]) for pdb in inputs ]
    for i, x in enumerate(r):
        x.get()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Finds close residues in the specified PDB structures.')
    parser.add_argument('inputs', nargs='+', type=str, help='PDB files or ids')
    parser.add_argument('--distance', type=float, default=4.0, help='Distance threshold between residue atoms (in Ångströms)')
    parser.add_argument('--output', type=str, help='An empty output directory (will create a parquet file for each PDB structure)', required=True)
    parser.add_argument("--pdbcache", nargs="+", help="Directories to search for PDB files in order to avoid re-downloading. Last directory will be written to, if the structure is not found and has to be downloaded from RCSB. Also can be specified as PDB_CACHE_DIR env variable.")
    parser.add_argument('--threads', type=para_utils.parse_thread_count, default=1, help='Number of threads, 0 for all, 50%% for half, -1 to leave one free, ... Parallelism only affects processing of multiple PDB files.')
    args = parser.parse_args()
    for x in args.pdbcache or []:
        pdb_utils.pdb_cache_dirs.append(os.path.abspath(x))
    os.environ["PDB_CACHE_DIR"] = ';'.join(pdb_utils.pdb_cache_dirs)

    if args.threads == 1:
        pool = para_utils.MockPool()
        _main(args, pool)
    else:
        with multiprocessing.Pool(args.threads if args.threads > 0 else max(1, os.cpu_count() + args.threads)) as pool:
            _main(args, pool)

