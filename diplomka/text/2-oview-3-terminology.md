## Base pair type terminology

### Saenger system {.unnumbered}

Since a large number of different non-canonical base pairs exists, a systematic way naming scheme is needed.
The first system was described by W. Saenger in the Principles of Nucleic Acid Structure book published in 1984 [TODO cite].
Seanger identified all possible base pairs with at least 2 hydrogen bonds and sorts them into 28 distinct classes.
The classes are identified only be an ordinal number, but ordered the classes 

TODO img https://www.nakb.org/basics/basepairs.html#Saenger/

![Seanger's ](https://www.nakb.org/basics/BPImages/saenger.gif)

### Leontis-Westhof system {.unnumbered}

A more systematic approach was proposed (by N. B. Leontis and E. Westhof in a paper published in 2001)[https://pubmed.ncbi.nlm.nih.gov/11345429].
They identified that each of the two nucleotides interacts with one of its three edges.
The "front edge" is named Watson-Crick, since it is the only one binding in the canonical base pairing.
One of the side edges is found binding in the Hoogsteen base pairing, so it is named Hoogsteen and shortened as H.
Note that the Hoogsteen base pairing is actually binding of a Hoogsteen edge onto a Watson-Crick edge.
The remaining edge is named Sugar, since this edge includes the covalent bond to ribose (N9-C1' or N1-C1').

![Nucleotide edges](https://www.nakb.org/basics/BPImages/base-edges.jpg)

A frequent misconception is that "Sugar" means the base binds to the ribose oxygen.
Although this is frequently the case that the base pair **includes** a hydrogen bond to the O2' atom, it is definitely not necessary.
The Sugar edge is primarily meant as one of purine/pyrimidine faces and most of the defined base pairs including the Sugar edge bind to an atom on the base, often the N3 purine atom.
<!-- The corner atoms are included in the definition of both edges -- for example, the N2 guanine atom is shared between the Sugar and Watson-Crick edges. ??? -->

TODO tSS GG

Some base pairs defined by Westhof and Leontis do bind exclusively to the O2' ribose atom.
However, this is the exception to the rule, and it makes us ask whether we shouldn't call these pairs "nucleotide pairs" instead of "base pairs".

A pair of edges still leaves two possible conformations as one of the nucleotides can be rotated by 180°.
Usually, this results in a different set of hydrogen bonds, so it is necessary to disambiguate the classes.
The Leontis-Westhof naming system calls one conformation **cis** and the other **trans**.
Cis essentially means that the N-C1' bonds point in the same direction (approximately, ± 180°).
Trans is the other option, the N-C1' bonds point in opposite directions.

Example: cWW GC, tWW GC

The named are commonly shortened to 3 letters - `c` or `t` for cis or trans, and `W`, `H` or `S` for each of the 2 edges. `W` is Watson-Crick, `H` Hoogsten and `S` is the Sugar edge.
Since the pairing conformation is different for each pair of base, we also include two `A`/`T`/`U`/`G`/`C` letters to identify them.
For example, we can say that the image above shows the `cWW GC` and `tWW GC` pairs.

### ?? {.unnumbered}

As far as we are away, all publications today use the Leontis-Westhof system.
In our opinion, the main disadvantage of the Saenger system is the need to remember the 28 classes.
The Leontis-Westhof system is also more general, of the 156 (??) possible classes we can observe 122 in high quality X-ray structures deposited in PDB. 

Being more general, the L-W system includes pairs which some might not want to call "base pairs".
A number of described base pairs only bind with a single hydrogen bonds or requires binding to ribose O2', restricting the class to RNA.
However, a few of doubly bonded legitimate base pairs are missing in the Saenger system, for example the cWw GG.



