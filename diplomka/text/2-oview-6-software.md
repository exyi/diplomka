## Software for base pair annotation

A number of software packages exist for determining which nucleotides are pairing, given a PDB or mmCIF structure.
Since there isn't any agreed upon definition of all basepair types, the different programs give slightly different results on most structures.
The differences are mostly constrained to non-typical basepairs and extreme forms of 

In this section, we'll list and briefly overview the available tools for basepair assignment.


### FR3D

We have already mentioned the FR3D program, since it is the main software we'll use for the analysis.
FR3D stands for “Find RNA 3D” and is commonly pronounced “Fred”.
[Author's original motivation was to find motifs of multiple nucleotides, we will not cover this in this work.](https://www.bgsu.edu/research/rna/software/fr3d.html)
First version of FR3D was written in Matlab, but it was later rewritten to Python.
We are using the Python version, downloaded from [github.com/BGSU-RNA/fr3d-python](https://github.com/BGSU-RNA/fr3d-python).
It seems that the Matlab version is not as actively maintained anymore.


The main advantages of FR3D are:

<!-- * It is freely available including the source code. -->
* TODO: convince Craig to actually license it?.
* It is easy to run and easy to process the output.
* It can annotate all pair families.
* It is currently maintained, we can thus influence its algorithms.

Unfortunately, FR3D does not support writing out basepair parameters

### X3DNA DSSR

X3DNA-DSSR is currently the most commonly used tool for the nucleic acid structure analysis.
X3DNA was first [published over twenty years ago](https://doi.org/10.1093/nar/gkg680), 3DNA stands for “3D Nucleic-acid Analysis”.
[DSSR, which stands for “Dissecting the Secondary Structure of RNA”](https://doi.org/10.1093/nar/gkv716) is a newer addition to the package specialized for RNA molecules with more complex structure.

DSSR is a commercial product that is not available for free for academic use at the time of writing.
[It must be licensed from the Columbia University at about $200 for the basic academic license.](https://inventions.techventures.columbia.edu/technologies/dssr-an-integrated--CU20391)
However, an older version, 1.9.9, was free for academic use to ensure reproducibility of the papers relying on DSSR analysis.
It was later withdrawn citing the lack of governmental funding as the reason for discontinuing the free distribution of DSSR.
On the other hand, the source code of [3DNA was made public in 2016](https://x3dna.org/highlights/3dna-c-source-code-is-available), although one must register on the X3DNA forum.

#### Basepair parameters

DSSR can calculate the standard base parameter discussed in [section](./2-oview-5-base-parameters.md).
To obtain the parameter values, we have to run DSSR with the `--analyze` flag.
It then writes out multiple files with the list of found pairs and their parameters.
One of the files also includes the translation vector and the orthonormal basis of the standard reference frame.

One of the DSSR limitations is that DSSR only writes out the basepair parameters for the pairs DSSR find in the structure.
For instance, we cannot use to complete this information to basepairs found by FR3D.
DSSR also has a `--json` which switches the output format to a machine-readable JSON, but this option unfortunately does not work with `--analyze`.


### RNAview

Similarly to FR3D, RNAview identifies basepairs the molecular structure and uses the naming Leontis-Westhof system.

https://github.com/rcsb/RNAView


### Curves+

https://doi.org/10.1093/nar/gkp608
