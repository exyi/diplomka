# Abbreviations and Conventions {.unnumbered}


## Conventions {.unnumbered}

For better clarity, we will use the following conventions, unless noted otherwise:

* To avoid the ambiguity of letters and numbers, PDB identifiers (codes) are written in lowercase letters, except for the letter L.
* If a figure of a molecule is not representative of its 3D structure, we use a chemical diagram. If it is planar projection of real 3D structure, we show a 3D render (from PyMOL).

## List of abbreviations {.unnumbered}

* **API** --- Application Programming Interface (an interface of a software library)
* **BGSU** --- Bowling Green State University
* **CIF** / **mmCIF** --- \[**m**acro**m**olecular\] Crystallographic Information File (standard file format for molecular structures)
* **CSV** --- Comma Separated Values (tabular text-based file format)
* **DDSR** --- "Dissecting the Spatial Structure of RNA" (nucleic acid analysis program, see @sec:software-DSSR)
* **FR3D** --- "Find RNA 3D" (nucleic acid analysis program, see @sec:software-FR3D)
* **JSON** --- JavaScript Object Notation (hierarchical text-based file format)
* **KDE** --- Kernel Density Estimate (non-parametric probability density estimator)
* **L-W** --- Leontis-Westhof naming system (explained in @sec:bp-terminology-lw)
* **NAKB** --- Nucleic Acid Knowledge Base
* **PCA** --- Principal Component Analysis
* **PDB** --- Protein DataBank (structural biology database; includes nucleic acids, despite the name)
* **RMSD** --- Root Mean Squared Deviation (structure similarity metric)
* **SVD** --- Singular Value Decomposition
* **SQL** --- Structured Query Language

**cWW**, **tWW**, **cWH**, **tWH**, **cWS**, **tWS**, **cHH**, **tHH**, **cHS**, **tHS**, **cSS**, **tSS** are abbreviated basepair families according to the Leontis-Westhof terminology. See @sec:bp-terminology-lw or <https://doi.org/10.1017/s1355838201002515> for explanation. Very common abbreviations such as DNA are omitted for brevity.

## Single letter abbreviations {.unnumbered}

Single letters are special category because of their ambiguity.
In this work, we use the letters **A**, **T**, **U**, **G**, **C** as the DNA and RNA bases --- adenine, thymine, uracil, guanine, and cytosine.
The letters **N**, **O**, **P**, **C**, **H** represent atom names.
**X**, **Y**, **Z** are the axes in three-dimensional space.

<!-- 
```
rg '\b[A-Z]{2,}\b' text/ --type md -o --no-filename --no-line-number | sort | uniq -c
``` -->
