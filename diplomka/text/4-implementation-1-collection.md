## Data collection script

The script `pair.py` calculates the distances between H-bonding atoms and other geometric parameters from mmCIF files downloaded from PDB.

### Input format

It does not determine which nucleotides are base-pairing by itself, this information must be provided as input.
Since we calculate the list of pairs using the fr3d-python package, the input format is identical to FR3D output.
For each mmCIF file, FR3D generates a `PDBID_basepair.txt` or a `PDBID_basepair_detailed.txt` file.
The text file contains a line for each determined basepair, formatted as UnitID space Family space UnitID.

The UnitID is a unique identified for nucleotides and other residues.
It is a structure PDB ID, Model Number, Chain ID, Nucleotide Number, and optionally Atom Name (not used in FR3D), Alternate ID, Insertion ID, and Symmetry Operation, separated by vertical bar character (`|`). [Detailed definition is at BGSU website](https://www.bgsu.edu/research/rna/help/rna-3d-hub-help/unit-ids.html)

Alternatively, the input file can be a table (CSV/Parquet) with columns `pdbid`, `model`, `chain1`, `res1`, `nr1`, `ins1`, `alt1`, `chain2`, `res2`, `nr2`, `ins2`, `alt2`.
Other columns are allowed and preserved on the output.
This data format provides compatibility with other software tools used at IBT.
<!-- TODO `--pair-type` option, rename to family? -->

### Produced information

The script loads all PDB structures from the input table using the [BioPython library](https://doi.org/10.1093/bioinformatics/btp163) and computes the following:

* Distance between heavy atoms of defined H-bonds
* Donor and acceptor angles
* Various metrics of coplanarity
* If the basepair is a dinucleotide
* If it is part of a parallel or antiparallel chain
* Adds Deposition Date, Structure Determination Method and Resolution to the output.

### Deduplication

FR3D reports all pairs twice, in both orientation.
For example, if a `cWH G-C` pair is reported, a corresponding `cHW C-G` is also reported.
Not to be too repetitive, we only want to work with one of the variants.
We have 3 rules to remove the duplicate entries.

1. If the pair family is asymmetrical, we keep the variant shown in LSW 2002 paper
    * Preferred families are `c`/`tWH`, `c`/`tWS`, `c`/`tHS`.
    * `HW`, `SW`, `SH` pairs are always dropped, or the order is swapped if the symmetrical one would not exist
2. If the pair nucleotides aren't equal, we keep the variant ordered according to `A > G > C > U`
    * For example, `cWW G-C` is preferred to `cWW C-G`
    <!-- * This means **A**denine is always the first
    * If **A**denine isn't present, **G**uanine must be first
    * Otherwise `C-U` is preferred to `U-C` -->
    * `T` is equivalent to `U`
3. Otherwise, the pair type name is completely symmetrical (`cWW G-G`)
    * We exclude the pair with longer H-bonds, if it is different.
    * If the H-bonds are also the same, we keep the pair with lower UnitID of the first nucleotide.

### X3DNA DSSR integration

Optionally, if `--dssr-binary` option is specified, basepair parameters columns are added.
However, as far as we know, it is not possible to force DSSR to compute basepair parameters for our selection of basepairs, it will determine the basepairing by itself and report the values.
Ths means that some basepairs might be missing the values.
Note that DSSR does recognize all basepair types reported by FR3D, in some runs all values might be missing (TODO specific example, also in ./2-oview-6-software.md).

The parameters are

* TODO

### Output format



TODO reklama na parquet?
