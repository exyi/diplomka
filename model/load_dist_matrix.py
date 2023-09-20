import Bio.PDB
import Bio.PDB.Structure
import Bio.PDB.Model
import os, sys, io, gzip, numpy as np

def _get_pdbid(file):
    pdbid = os.path.basename(file).split(".")[0]
    assert len(pdbid) == 4
    return pdbid

def load_pdb(pdb_or_file, pdb_id = None) -> Bio.PDB.Structure.Structure:
    if pdb_or_file is None and pdb_id is not None:
        # download from https://files.rcsb.org/download/XXXX.cif.gz
        import urllib.request
        url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
        with urllib.request.urlopen(url) as response:
            with gzip.open(io.BytesIO(response.read())) as f:
                return load_pdb(f, pdb_id)

    elif isinstance(pdb_or_file, str):
        if pdb_or_file.endswith(".gz"):
            with gzip.open(pdb_or_file) as f:
                return load_pdb(f, pdb_id or _get_pdbid(pdb_or_file))
        elif pdb_or_file.endswith(".zst"):
            import zstandard
            with zstandard.open(pdb_or_file) as f:
                return load_pdb(f, pdb_id or _get_pdbid(pdb_or_file))
        else:
            with open(pdb_or_file) as f:
                return load_pdb(f, pdb_id or _get_pdbid(pdb_or_file))
            
    else:
        parser = Bio.PDB.MMCIFParser()
        structure = parser.get_structure(pdb_id, pdb_or_file)
        return structure

def get_atom_df(model: Bio.PDB.Model.Model):
    atom_count = sum(1 for atom in model.get_atoms())
    atom_res = np.zeros(atom_count, dtype=np.int32)
    atom_resname = np.zeros(atom_count, dtype="S4")
    atom_chain = np.zeros(atom_count, dtype="S4")
    atom_coord = np.zeros((atom_count, 3), dtype=np.float32)
    atom_element = np.zeros(atom_count, dtype="S4")
    atom_name = np.zeros(atom_count, dtype="S4")

    atom_idx = 0
    for chain in model.get_chains():
        for residue in chain.get_residues():
            for atom in residue.get_atoms():

                atom_res[atom_idx] = residue.id[1]
                atom_resname[atom_idx] = residue.resname
                atom_chain[atom_idx] = chain.id
                atom_coord[atom_idx, :] = atom.coord
                atom_element[atom_idx] = atom.element or ""
                atom_name[atom_idx] = (atom.name or '').encode('utf-8')

                atom_idx += 1

    return {
        "chain": atom_chain,
        "res": atom_res,
        "resname": atom_resname,
        "coord": atom_coord,
        "element": atom_element,
        "name": atom_name,
    }

