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
* Various metrics of coplanarity (see TODO)
* If the basepair is a dinucleotide (i.e. are covalently bonded though the phosphodiester bond)
* If it is part of a parallel or an antiparallel chain
* Adds the Structure Deposition Date, Determination Method and Resolution metadata to the output.

### Deduplication

FR3D reports each pairs twice, in both orientations.
For illustration, if a `cWH G-C` pair is reported, a corresponding `cHW C-G` pair is also reported.
To avoid redundancy, we will de-duplicate the outputs using the following three rules:

1. If the pair family is asymmetrical, we keep the variant shown in LSW 2002 paper
    * Preferred families are cis or trans `WH`, `WS`, `HS`.
    * `HW`, `SW`, `SH` pairs are always dropped
2. If the pair nucleotides aren't equal, we keep the variant ordered according to `A > G > C > U`
    * For instance, `cWW G-C` is preferred to `cWW C-G`, as `C` is before `G` in the ordering
    * `T` is treated as equivalent to `U`
3. Otherwise, the pair type name is completely symmetrical (`cWW G-G`)
    * We exclude the pair with longer H-bonds, if it is different.
    * If the H-bonds are also the same, we keep the pair with lower UnitID of the first nucleotide.

### X3DNA DSSR integration

Optionally, if `--dssr-binary` option is specified, the basepair parameters computed by DSSR are added (see @sec:std-base-parameters).
However, as far as we know, it is not possible to force DSSR to compute basepair parameters for an arbitrary selection of basepairs, it will only report the parameters for basepairs it determined by itself.
This unfortunately means that some basepairs might be missing the parameters.
Although DSSR should recognize all basepairs types reported by FR3D, sometimes almost all the parameter values are missing (TODO specific example, also in ./2-oview-6-software.md).

The rationale for executing DSSR within the `pairs.py` script, instead of running it on all structures beforehand like we do with FR3D, lies in the complexity of DSSR's output format.
While FR3D generates a single "PDBID_basepair.txt" file, DSSR generates a collection of files such as "dssr-dsStepPars.txt", "dssr-dsStepPars.txt", "dssr-dsHelixPars.txt", and "dssr-basepairs.txt".
The fact that the output filenames cannot be easily changed prevents us from simply running DSSR in a loop for all structures.
Since the outputs contain a lot of values, the files are significantly larger than those produced by FR3D.

DSSR provides a machine-readable JSON output format using the `--json` option, but this option does nothing alongside the `--analyze` option.
Since DDSR only calculates the base parameters when the `--analyze` is specified, we have to parse the values from the loosely formatted text files.

### Output format

The output is a single table with a row for each basepair and a column for each computed parameter.
We'll leave detailed schema description for Appendix - Data Schemas TODO.
The `pairs.py` script outputs two files - a CSV table and an equivalent Parquet table.
In further processing, we currently exclusively use the Parquet file, but the CSV format arguably offers easier integration with any other scripts.

#### Parquet format

Parquet is a modern binary format for tables, usually praised for its better performance compared to alternatives.
Data in the file is organized 
Our main reason for choosing the format it 
