## Software for base pair annotation

A number of software packages exist for determining which nucleotides are pairing, given a PDB or mmCIF structure.
Since there isn't any agreed upon definition of all basepair types, the different programs give slightly different results on most structures.
The differences are mostly constrained to the basepairs where the conformance to the definition is questionable, for example when only one H-bond

In this section, we'll list and briefly compare other available tools.


### FR3D

We have already mentioned the FR3D program, since it is the main software we'll use for the analysis.
The main advantages are
<!-- * permisive license TODO: force craig to actually license it? -->
* it is easy to run and parse the output
* it can annotate all the pairs
* it is currently maintained

FR3D stands for “Find RNA 3D” and is commonly pronounced “Fred”.
Author's original motivation was to find motifs of multiple nucleotides, which we'll not cover in this work. [https://www.bgsu.edu/research/rna/software/fr3d.html]
First version of FR3D was written in Matlab, but was later rewritten to Python.
We are using the Python version, downloaded from [github.com/BGSU-RNA/fr3d-python](https://github.com/BGSU-RNA/fr3d-python).
The Matlab version is not as actively maintained anymore.

### X3DNA DSSR


