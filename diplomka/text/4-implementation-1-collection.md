## Data Collection Script

The script `pair.py` calculates the distances between H-bonding atoms and other geometric parameters from mmCIF files downloaded from PDB.

### Input Format

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

### Produced Information

The script loads all PDB structures from the input table using the [BioPython library](https://doi.org/10.1093/bioinformatics/btp163) and computes the following:

* The measured parameters, as defined in @sec:basepair-metrics
    * Distance between heavy atoms of defined H-bonds
    * Donor and acceptor angles
    * Various metrics of coplanarity and relative base orientation
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

### Pairs between asymmetrical unit {#sec:impl-collection-symmetry}

Crystals are made of the same molecule, repeated many times, but they not necessarily always in the same orientation.
Crystallographers have a comprehensive theory for describing these repetitions, it is crucial for resolving the molecular structures from diffraction patterns.
The X-Ray structures in PDB only contain one of the repeating fragments — the **asymmetrical unit**.
If the molecule of interest forms a symmetrical dimer, the interactions between asymmetrical units are potentially relevant for biology.
In nucleic acids, double helices and tetraplexes may be completely symmetrical, in which case we have to work with the same structure in two copies.

![The [`6ros`](https://www.rcsb.org/structure/6ROS) structure only has coordinates of one of the chains, the second one is a symmetric copy. All basepairs are between the two.](../img/6ros-symmetry-illustration.png)

PDBx/mmCIF files include the complete description of the crystal symmetry as the crystal space group.
Helpfully, the files also encode the biologically relevant symmetry operation as [a rotation matrix and a translation in the `pdbx_struct_oper_list` CIF category](https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Categories/pdbx_struct_oper_list.html).
This category may contain any number of transformations, labeled by the [PDB symmetry operation code](http://www.bmsc.washington.edu/CrystaLinks/man/pdb/part_74.html).
We can thus avoid the space group operation calculations and rely on the provided transformation.

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

Technically, the implementation is still slightly tricky, because the **BioPython** library does not parse the `pdbx_struct_oper_list` category.
Another Python library with similar API -- **Gemmi**, has a very good support for crystallographic symmetry.
It, however, exposes the information in terms of space groups, instead of the PDB symmetry operation codes.
Since **FR3D** uses the PDB codes in its output, we need to use them to map the basepairs into atomic coordinates.
We thus use another library -- **mmcif**, which simply parses the CIF without any additional abstraction.

PyMOL has a direct support for assembling the biological unit, the [structure only has to be loaded after setting `assembly` flag to 1](https://pymolwiki.org/index.php/Assembly) (see @sec:impl-basepair-img-asy for more details on PyMOL usage).

### X3DNA DSSR integration

Optionally, the basepair parameters computed by DSSR are added (see @sec:std-base-parameters), `pairs.py` runs DSSR when `--dssr-binary` option is specified.
However, as far as we know, it is not possible to force DSSR to compute basepair parameters for an arbitrary selection of basepairs, it will only report the parameters for basepairs it determined by itself.
This unfortunately means that some basepairs might be missing the parameters.
Although DSSR should recognize all basepairs types reported by FR3D, sometimes almost all the parameter values are missing (TODO specific example, also in ./2-oview-6-software.md).

The rationale for executing DSSR within the `pairs.py` script, instead of running it on all structures beforehand like we do with FR3D, is the complexity of DSSR's output format.
While FR3D generates a single "PDBID_basepair.txt" file, DSSR generates a collection of files such as "dssr-dsStepPars.txt", "dssr-dsStepPars.txt", "dssr-dsHelixPars.txt", and "dssr-basepairs.txt".
The fact that the output filenames cannot be easily changed prevents us from simply running DSSR in a loop for all structures.
Since the outputs contain a lot of values, the files are significantly larger than those produced by FR3D.

DSSR provides a machine-readable JSON output format using the `--json` option, but this option does nothing alongside the `--analyze` option.
Since DDSR only calculates the base parameters when the `--analyze` is specified, we have to parse the values from the loosely formatted text files.

### Output Format

The output is a single table with a row for each basepair and a column for each computed parameter.
We'll leave detailed schema description for Appendix — Data Schemas TODO.
The `pairs.py` script outputs two files — a CSV table and an equivalent Parquet table.
In further processing, we currently exclusively use the Parquet file, but the CSV format arguably offers easier integration with any other scripts.

#### Parquet

[Apache Parquet](https://en.wikipedia.org/wiki/Apache_Parquet) is a modern binary format for tabular data, comparable to CSV.
Usually, it is praised for being fast to process, but the main reason for choosing the format in this work is the typed schema.
Parquet, compared to CSV, specifies the type of each column — integer / decimal number / text.
This might seem like a marginal issue, in most cases it is easy to infer the type from the data — we see if the column contains letters or only number.
However, it might lead to crashes on empty files, or on short files which coincidentally have valid numbers in the text column.
For instance, grouping the entries by the PDB structure causes this problem, since the PDB Identifiers may be valid numbers in the exponential format (`5e95` = $5\times10^{95}$).
In this example, we don't get a crash, but the DuckDB database formats the number differently when converting it back to text, leading to other issues.

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

Parquet is complicated data format with many features, including support for complex data types — structures and arrays.
Even though the hydrogen bond parameters could benefit from it, we choose to keep the Parquet and CSV files equivalent and avoid using the complex data types.
Instead of one array of structures `{ length: Float64, donor_angle: Float64, ... }`, we have multiple columns `hb_0_length`, `hb_0_donor_angle`, ..., `hb_1_length`, `hb_1_donor_angle`, and so on.
Since we never have more than four defined bonds, it is not a significant issue.

A standardized method for handling nested columns in CSV files would be beneficial, perhaps through a consistent syntax such as `structure_field.nested_field`.
This would allow us to leverage Parquet structs to better organize the large number of columns present in our output files while keeping consistency with the CSV output.
Today, most tools simply refuse to create a CSV from such table with structures, so we avoid the feature even for this use case.


#### Data Schema {#sec:data-collection-schema}

TODO: should be appendix??
