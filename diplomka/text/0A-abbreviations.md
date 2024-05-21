# Abbreviations and Conventions {.unnumbered}


## Conventions {.unnumbered}

For better clarity, we will use the following conventions, unless noted otherwise:

* PDB identifiers (codes) are written in lowercase letters, except for the letter L, which is always written in uppercase.
* If a figure of a molecule isn't representative of its 3D structure, a chemical diagram is used. If it is planar projection of real 3D structure, a 3D rendered image (from PyMOL) is used.

## List of Abbreviations {.unnumbered}

For brevity, the list only includes ambiguous or uncommon abbreviations.

* **TODO** — this means that the part of the work is incomplete

**cWW**, **tWW**, **cWH**, **tWH**, **cWS**, **tWS**, **cHH**, **tHH**, **cHS**, **tHS**, **cSS**, **tSS** are abbreviated basepair families. See @sec:bp-terminology-lw or [](https://doi.org/10.1017/s1355838201002515) for explanation.

### Single Letter Abbreviations {.unnumbered}

Single letters are special category because of their ambiguity.
In this work, we use the letters **A**, **T**, **U**, **G**, **C** for the DNA and RNA bases -- adenine, thymine, uracil, guanine and cytosine.
The letters **N**, **O**, **P**, **C**, **H** are atom names.
**X**, **Y**, **Z** are the three axes in three-dimensional space.


```
rg text/ --type md '\b[A-Z]{2,}\b' -o --no-filename --no-line-number | sort | uniq -c
```
