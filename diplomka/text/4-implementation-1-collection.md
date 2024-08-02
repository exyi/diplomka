## Data Collection Script — `pairs.py` {#sec:sw-collection}

The script `pairs.py` analyzes all classes of basepairs and computes the geometric parameters (@sec:basepair-params) from mmCIF files downloaded from [the RCSB PDB](https://doi.org/10.1093/nar/28.1.235).

### Input format

This program does not determine which nucleotides are base-pairing by itself; this information must be provided as input.
Since we calculate the list of pairs using the fr3d-python package, the input format is identical to FR3D output.
For each mmCIF file, FR3D generates a `PDBID_basepair.txt` or a `PDBID_basepair_detailed.txt` file.
The text file contains a line for each determined basepair, formatted as "**UnitID** tab **Family** tab **UnitID**".

The UnitID is a unique identifier for nucleotides and other residues --- it is a structure PDB ID, Model Number, Chain ID, Nucleotide Number, and optionally Atom Name (not used in FR3D), Alternate ID, Insertion ID, and Symmetry Operation.
The components are separated by a vertical bar character (`|`).
[Detailed definition is at BGSU website](https://www.bgsu.edu/research/rna/help/rna-3d-hub-help/unit-ids.html)

Alternatively, the input file may be a table (CSV/Parquet) with columns `pdbid`, `model`, `chain1`, `res1`, `nr1`, `ins1`, `alt1`, `chain2`, `res2`, `nr2`, `ins2`, `alt2`, and `family`.
Other columns are allowed and preserved on the output.
This data format provides compatibility with other software tools used at IBT.
The same columns identify the basepairs in the output files.

### Produced information

The script loads all PDB structures from the input table using the [BioPython library](https://doi.org/10.1093/bioinformatics/btp163) and computes the following:

* The measured parameters, as defined in @sec:basepair-params
    * Distance between heavy atoms of defined H-bonds
    * Donor and acceptor angles
    * The coplanarity metrics (distances and angles between base planes)
    * Relative base orientation --- the yaw, pitch, and roll angles and also the relative translation
* An `is_dinucleotide` column indicating if the basepair is a di-nucleotide (i.e., are covalently bonded through the phosphodiester bonds)
* An `is_parallel` column indicating if the pair is part of a parallel or an antiparallel chain.
* Structure Name (`structure_name`), Determination Method (`structure_method`), Structure Deposition Date (`deposition_date`), and Resolution (`resolution`) metadata columns.

### Output format

The output is a single table with a row for each basepair and a column for each computed parameter.
The `pairs.py` script outputs two files — a CSV table and an equivalent Parquet table.
In further processing, we currently exclusively use the Parquet file, but the CSV format offers easier integration with third-party software.

#### Parquet

[Apache Parquet](https://en.wikipedia.org/wiki/Apache_Parquet) is a modern binary format for tabular data, comparable to a compressed CSV.
Usually, Parquet is praised for being fast to process; for us, the main reason for choosing the format is the strongly-typed schema.
Parquet, compared to CSV, specifies the data type of each column — integer / floating-point number / text.
At first, it might seem easy to infer the data type from the CSV data — we can see if the column contains letters or only digits.
However, the inference will inevitably fail on empty tables, or on short files which coincidentally solely have valid numbers in a text column.
For instance, grouping the entries by the PDB structure causes the latter issue, since the PDB Identifiers may be valid numbers in the exponential format (`5e95` = $5\times10^{95}$).
The DuckDB database then formats the number differently when converting it back to text, leading to issues further down the line.

```sql
DuckDB> select pdbid from './1e95.csv' union
        select pdbid from './1ezn.csv';
┌─────────┐
│  pdbid  │
│ varchar │
├─────────┤
│ 1ezn    │
│ 1e+95   │
└─────────┘
```

Parquet is a complex data format with many features, including support for composite data types — structures and arrays.
However, we choose to avoid using composite data types to keep the Parquet and CSV files equivalent, even though the hydrogen bond parameters could benefit from it.
Instead of an array of structures `{ length: Float64, donor_angle: Float64, ... }`, we have multiple columns `hb_0_length`, `hb_0_donor_angle`, …, `hb_1_length`, `hb_1_donor_angle`, ….
Since we never have more than four defined bonds, it is not a significant issue.

We use the [Polars DataFrame library](https://doi.org/10.5281/zenodo.7697217) in Python code and [DuckDB-wasm](https://github.com/duckdb/duckdb-wasm) in the web application as the Parquet reader and writer, and we can recommend both implementations.

### Deduplication

FR3D reports each pair twice, in both orientations --- if a **cWH G-C** pair is reported, a corresponding **cHW C-G** pair is also reported.
To avoid redundancy, we deduplicate the pair using the following rules:

1. If the pair family is asymmetric, we keep the variant shown in <https://doi.org/10.1093/nar/gkf481>.
    * The referred families are **WH**, **WS**, **HS** (cis or trans),
    * while **HW**, **SW**, **SH** pairs are always dropped.
2. If FR3D lowercased the first family edge letter (@sec:bp-fr3d-lowercasing), we drop the pair.
3. If the pair nucleotides are not equal, we keep the variant ordered according to the **A** > **G** > **C** > **U** = **T** rule.
    * For instance, **cWW G-C** is preferred to **cWW C-G**, as `C` is before `G` in the ordering.
4. Otherwise, the pair type name is completely symmetric (`cWW G-G`).
    * We exclude the pair with longer H-bonds.
    * If the H-bonds are the same, we keep the pair with lower UnitID of the first nucleotide.

### Pairs between symmetry-related nucleotides {#sec:impl-collection-symmetry}

Asymmetric unit, the smallest part of the crystal [from which the whole crystal can be re-built](isbn:978-0815340812), does not in all crystal structures contain the whole biologically relevant <!-- molecule or --> molecular complex.
As an example, double-helical DNA or RNA of a palindromic sequence can have only one nucleotide strand in the asymmetric unit while the biologically relevant structure is the duplex.
Because PDB files contain coordinates only for one asymmetric unit we have to consider the possibility for the basepair assignment across the symmetry operation as two bases forming the pair can be symmetry-related. 

![The asymmetric unit of [`6ros`](https://www.rcsb.org/structure/6ROS) structure is formed by a single strand, but the **biological assembly** is a duplex. The mmCIF file contains the coordinates of only one strand, and the second one is its symmetric copy. All basepairs are formed between the two strands.](../img/6ros-symmetry-illustration.png){#fig:6ros-symmetry-illustration}

PDBx/mmCIF files include the information to complete the biological unit as [a rotation matrix and a translation vector in the `pdbx_struct_oper_list` mmCIF category](https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Categories/pdbx_struct_oper_list.html).
This category may contain any number of transformations, labeled by the [PDB symmetry operation code](http://www.bmsc.washington.edu/CrystaLinks/man/pdb/part_74.html).
We thus do not have to handle the space group operations and solely rely on the provided transformation matrices.

<!-- ```
loop_                                                                                                                          
_pdbx_struct_oper_list.id                                     
_pdbx_struct_oper_list.type                                                                                                    
_pdbx_struct_oper_list.name
_pdbx_struct_oper_list.symmetry_operation                                                                                      
_pdbx_struct_oper_list.matrix[1][1]
_pdbx_struct_oper_list.matrix[1][2]                                                                                            
_pdbx_struct_oper_list.matrix[1][3]
_pdbx_struct_oper_list.vector[1]                                                                                               
_pdbx_struct_oper_list.matrix[2][1]
_pdbx_struct_oper_list.matrix[2][2]                                                                                            
_pdbx_struct_oper_list.matrix[2][3]
_pdbx_struct_oper_list.vector[2]                                                                                               
_pdbx_struct_oper_list.matrix[3][1]
_pdbx_struct_oper_list.matrix[3][2]                                                                                            
_pdbx_struct_oper_list.matrix[3][3]
_pdbx_struct_oper_list.vector[3]                                                                                               
1 'identity operation'         1_555 x,y,z      1.0000000000 0.0000000000 0.0000000000 0.0000000000   0.0000000000 1.0000000000
0.0000000000 0.0000000000  0.0000000000 0.0000000000 1.0000000000  0.0000000000                                     
2 'crystal symmetry operation' 7_465 y-1,x+1,-z 0.0000000000 1.0000000000 0.0000000000 -38.4400000000 1.0000000000 0.0000000000
0.0000000000 38.4400000000 0.0000000000 0.0000000000 -1.0000000000 0.0000000000
``` -->

Technically, the implementation is still slightly tricky because we are aware of **BioPython** API to access the `pdbx_struct_oper_list` category.
Another Python library --- [**Gemmi**](https://doi.org/10.21105/joss.04200), has a very good support for crystallographic symmetry.
It, however, exposes the information in terms of space groups, instead of the PDB symmetry operation codes.
Since **FR3D** uses the PDB codes in its output, we need to use these to map the basepairs into atomic coordinates.
Therefore, we utilize the [**mmcif**](https://github.com/rcsb/py-mmcif) library to get this information.
As **mmcif** simply parses the CIF without any additional abstraction, we currently use it in addition to **BioPython**.

[PyMOL](https://github.com/schrodinger/pymol-open-source) has a direct support for assembling the biological unit, the [structure only has to be loaded after setting `assembly` flag to 1](https://pymolwiki.org/index.php/Assembly) (more details is in @sec:impl-basepair-img-asy).

<!-- ### X3DNA DSSR integration

Optionally, the basepair parameters computed by DSSR are added (see @sec:std-base-parameters), `pairs.py` runs DSSR when `--dssr-binary` option is specified.
However, as far as we know, it is not possible to force DSSR to compute basepair parameters for an arbitrary selection of basepairs, it will only report the parameters for basepairs it determined by itself.
This unfortunately means that some basepairs might be missing the parameters.
Although DSSR should recognize all basepairs types reported by FR3D, sometimes almost all the parameter values are missing (xxxxxxxxxxxxxxxxxx specific example, also in ./2-oview-6-software.md).

The rationale for executing DSSR within the `pairs.py` script, instead of running it on all structures beforehand like we do with FR3D, is the complexity of DSSR's output format.
While FR3D generates a single "PDBID_basepair.txt" file, DSSR generates a collection of files such as "dssr-dsStepPars.txt", "dssr-dsStepPars.txt", "dssr-dsHelixPars.txt", and "dssr-basepairs.txt".
The fact that the output filenames cannot be easily changed prevents us from simply running DSSR in a loop for all structures.
Since the outputs contain a lot of values, the files are significantly larger than those produced by FR3D.

DSSR provides a machine-readable JSON output format using the `--json` option, but this option does nothing alongside the `--analyze` option.
Since DDSR only calculates the base parameters when the `--analyze` is specified, we have to parse the values from the loosely formatted text files. -->


### Partitioning and parallelism

By default, the `pairs.py` script loads all basepairs into memory to group them by PDB ID and subsequently process each structure individually.
This enables straightforward parallel processing using the Python `multiprocessing` module: we group the table of pairs by PDB ID and use `multiprocessing.Pool.map(...)` to process each structure.
In an effort to maintain uniform CPU time consumption and to avoid mean emails from the PBS daemon, we split structures with more than 100 000 basepairs into several groups, and prioritize larger groups over smaller ones.

In the context of annotating basepairs on the entire PDB (@sec:testing-basepair-params), we must compute the parameters for all _basepair candidates_, i.e., pairs of nucleotides in proximity to one another.
Given the substantial size of the dataset, it is practical to subdivide it into smaller chunks and process them sequentially.
This functionality can be invoked using the `--partition-input-select=K/N` command-line option, where **N** is a placeholder for the total number of partitions, and **K** is the specific partition to be processed in the current run.
The parameter `K` can also be specified to be a range, allowing for execution of multiple partitions via a single command, such as `--partition-input-select=0-64/64`.
Apart from reducing memory usage, the option may be used for parallel processing across multiple machines.
