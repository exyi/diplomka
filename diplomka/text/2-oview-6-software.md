## Software for Base Pair Annotation {#sec:software}

Several software packages exist for determining which nucleotides are pairing within a given PDB or mmCIF structure.
However, due to the absence of universally agreed-upon definitions for all basepair types, different programs usually give slightly different results.
<!-- The differences are mostly constrained to non-typical basepairs and near  -->

In this section, we will list and briefly overview the available tools used for basepair assignment.


### FR3D {#sec:software-FR3D}

We have previously introduced FR3D, which will serve as our primary analysis tool.
FR3D stands for “Find RNA 3D” and is pronounced “Fred”.
[Initially, the software was developed to identify motifs comprised of multiple nucleotides](https://www.bgsu.edu/research/rna/software/fr3d.html).
We will not cover that extensively, but it is relevant because [the motif atlas](https://doi.org/10.1261/rna.039438.113) published by the FR3D authors relies on the determined basepairs as its source data.

First version of FR3D was written in MATLAB, but it was later rewritten to Python.
We are using the Python version, downloaded from [github.com/BGSU-RNA/fr3d-python](https://github.com/BGSU-RNA/fr3d-python){.link-no-footnote}.
It appears that the MATLAB version is not as actively maintained anymore.

The primary advantages of FR3D for our use include:

* It is freely available including the source code licensed under [the Apache 2 license](https://github.com/BGSU-RNA/fr3d-python/blob/master/pyproject.toml#L10).
* Ability to annotate all basepair families according to the L-W system.
* Basepair determination quality.
* Active maintenance, allowing us to potentially influence its algorithms.
* Ease of use in execution and output processing.

FR3D does not calculate the standard basepair parameters discussed in @sec:std-base-parameters, and it does not support writing out its own calculated parameters.
Because of that, we would have to modify FR3D or use other software to calculate them.
FR3D determines the basepairs based on a set of custom rules, distinct for each type of basepairs.
The rules are not described in literature or documentation, but the rules are conveniently placed in a single source file: [`classifiers/class_limits.py`](https://github.com/BGSU-RNA/fr3d-python/blob/6f0a75ed547c7862d804a8a70ad73e04de89955f/fr3d/classifiers/class_limits.py#L340){.link-no-footnote}.

Thanks to the large list of handcrafted rules, FR3D is able to rule out many edge cases while also including the non-standard basepairs.
It has been validated on a [representative set of PDB structures](https://doi.org/10.1007/978-3-642-25740-7_13) by its authors and thoroughly compared against other base determination software.
The cross-validation experiments are currently available online at [rna.bgsu.edu](http://rna.bgsu.edu/experiments/annotations/compare_v7_cWW_A,G_3.0A.html); the URL may be changed to examine any basepair type.

### X3DNA DSSR {#sec:software-DSSR}

X3DNA-DSSR is currently the most commonly used tool for nucleic acid structure analysis.
X3DNA was first [published over twenty years ago](https://doi.org/10.1093/nar/gkg680), with “3DNA” standing for “3D Nucleic-acid Analysis”.
[DSSR, which stands for “Dissecting the Secondary Structure of RNA”](https://doi.org/10.1093/nar/gkv716) is a newer addition to the package specialized for RNA molecules with more complex structure.

DSSR is a commercial product that is not available for free at the time of writing.
[It requires licensing from Columbia University at about \$200 for a basic academic license.](https://inventions.techventures.columbia.edu/technologies/dssr-an-integrated--CU20391)
Conveniently, an older version, 1.9.9, is free for academic use to ensure reproducibility of the literature relying on DSSR analysis.
The free version was later withdrawn citing the lack of governmental funding as the reason.
On the other hand, the source code of [3DNA was made public in 2016](https://x3dna.org/highlights/3dna-c-source-code-is-available), although users are required to register on the X3DNA forum.

<!-- Given unavailability its source code, we do not know what exactly are the base determination rules in DSSR.
It appears to be a set of rules based on the standard basepair parameters. -->

#### Basepair parameters {#sec:software-DSSR-basepair-params}

DSSR is capable of calculating standard base parameters as discussed in @sec:std-base-parameters.
To obtain the parameter values, DSSR must be executed with the `--analyze` flag, which results in the generation of multiple files containing lists of identified pairs and their calculated parameters.
One of the produced files also includes the translation vector and the orthonormal basis of each basepair's standard reference frame.

A limitation of DSSR is that it exclusively reports basepair parameters for the pairs identified within the input structure.
In other words, it cannot provide complementary information to basepairs identified by alternative methods, such as FR3D.


### RNAView {#sec:software-RNAview}

Similarly to FR3D, [RNAView](https://doi.org/10.1093/nar/gkg529) identifies basepairs the molecular structure and uses the naming L-W system.
First published in 2003, it has received recent activity in development; notably, it has got support for mmCIF files a few months ago.
[The current version is published on GitHub](https://github.com/rcsb/RNAView) and available under the Apache 2 license.

Somewhat uniquely, [Yang et al.](https://doi.org/10.1093/nar/gkg529) detail the used basepair determination algorithm.
In short, they consider three basic rules:

1. The angle between base planes must be below 65°.
2. The vertical distance between the planes (at the point of contact) must be below 2.5 Å.
3. Two hydrogen bonds must exist, one of which must be polar and shorter than 3.4 Å.

However, RNAView unfortunately does not perform that well in practice; although we find it inspiring that such simplicity is feasible.
While the algorithm may at first feel to be quite lax, RNAView has more problems identifying good basepairs than misidentifying bad ones.
For instance, in the **tHS A-C** class, RNAView frequently misidentifies pairs as another family or misses them entirely --- we can see that [on BGSU experimental annotation comparison](http://rna.bgsu.edu/experiments/annotations/compare_v7_tHS_A,C_3.0A.html).

![A clear trans Hoogsteen/Sugar A-C basepair as trans Watson/Sugar basepair misidentified by RNAView. The mistake is very understandable, as the adenine is bound exclusively through the N6 atom, which is part of both Hoogsteen and Watson-Crick edges. See @sec:basepair-params-ypr for more discussion regarding this issue.](../img/rnaView-tHS-AC-misidentified.png){#fig:rnaView-tHS-AC-misidentified .img-width75}


### Curves+ {#sec:software-Curves}

[Curves](https://doi.org/10.1093/nar/gkp608) is an older software written in Fortran, but it has been recently (2016) updated, with the updated version named **Curves+**.
Despite that, we did not find a working web server nor documentation on how to use the program.
We have the Fortran source code and the binary executable, but it also appears to have disappeared from the internet.

Curves+ is the second software tool which can calculate the standard basepair parameters (@sec:std-base-parameters), and [its publication](https://doi.org/10.1093/nar/gkp608) details how to achieve symmetry of the parameters.
The values computed by Curves+ should be equal, regardless if it encounters one base first, or the other.
The trick is to get an average of the two reference frames and then consider the relative position of the bases from the average.
The rotations or translations of the two bases can then simply be added, getting the total rotation between them.


<!-- The original Curves was the subject of discussion in the ["Resolving the discrepancies among nucleic acid conformational analyses"](https://doi.org/10.1006/jmbi.1998.2390), since it used different reference frame and different formulas for the parameters.
However, Curves+ resolves the issue, allowing the standard reference frame -->
