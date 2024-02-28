## Basepair type terminology

### Saenger system

Since many different non-canonical base pairs exists, a systematic way naming scheme is needed.
The first naming scheme we know about was described by [W. Saenger in the Principles of Nucleic Acid Structure book published in 1984](https://doi.org/10.1007/978-1-4612-5190-3).
W. Seanger identified possible base pairs with at least 2 hydrogen bonds and sorted them into 28 distinct classes.

The classes are identified by a single ordinal number, and are sorted into five higher-level according to geometrical symmetry, and the pairing bases.


![Seanger's system of base pair types, [from the](https://doi.org/10.1007/978-1-4612-5190-3) book, page 120](../img/saenger-system-merged.png)

### System of three edges

A more systematic approach was proposed [by N. B. Leontis and E. Westhof in a paper published in 2001](https://doi.org/10.1017/s1355838201002515).
The authors identified that each of the two nucleotides interacts with one of its three edges.
The “front edge” is called **“Watson-Crick”**, since this is the only edge involved in the canonical base pairing.
The two side edges are called **“Hoogsteen”** and **“Sugar”** -- the former is involved in the Hoogsteen pairs and the latter covalently bonds with the ribose sugar.
The base pairs are named by the two interacting edges, and their relative orientation (cis, trans).

![The three edges of a pyrimidine (U) and a purine (G)](../img/purine-pyrimidine-edges.svg)

A pair of edges still leaves two possible conformations -- either the indicated edge arrows point in opposite directions, or they are parallel if we flip one of the bases.
Usually, this results in a different set of hydrogen bonds, and it will require different shape of the nucleic acid backbone.
The Leontis-Westhof naming system calls one conformation **cis** and the other **trans**, based on the direction of the N-C1' bonds.
Generally, **cis** pairs have the N-C1' bonds pointing in a similar direction, while **trans** pairs orient them in the same direction.
However, there isn't any specific 90° limit on their angle, we instead label **cis** the conformation which is more acute angle than the **trans** conformation.
There exist basepairs where both angles can be above 90°, albeit only slightly in case the **cis** conformation.

The **cis** and **trans** prefixes only indicate which way the bases face each other, it does not indicate if the pairing helix is parallel or antiparallel, although it correlates.
Canonical basepairs are **cis** Watson-Crick G-C and A-T with antiparallel strands.
For instance, **trans** Watson-Crick G-G also occurs in antiparallel strands (see the [`4pmi`](https://www.rcsb.org/structure/4pmi) PDB structure).
The [“Strand Orientation” table at NAKB](https://www.nakb.org/basics/basepairs.html#LW/) is thus not entirely correct, it is only indicating the typical case. 

<!-- **Cis** basepairs have the pairing edge arrows oriented in parallel, while **trans** basepairs have edges in opposing direction.

However, to pair in **cis**, the bases must be flipped -->

Note that “Sugar” is a name of the purine or pyrimidine edge -- it does not necessarily mean that the pairing base is interacting with the ribose sugar, although it is frequently the case.
In RNA, we often find the ribose O2' oxygen interacting with the other base, or even the other O2' oxygen.
Since it is an OH group, it can act either as both an H-bond donor and an acceptor.

TODO Show example image of tSS GG + something that needs the O2'

![](../img/tSS-GG-DNA-6n4g-A_2-B_2-no-oxygens.png)

<!-- A frequent misconception is that "Sugar" means the base binds to the ribose oxygen.
Although this is frequently the case that the base pair **includes** a hydrogen bond to the O2' atom, it is definitely not necessary.
The Sugar edge is primarily meant as one of purine/pyrimidine faces and most of the defined base pairs including the Sugar edge bind to an atom on the base, often the N3 purine atom.
The corner atoms are included in the definition of both edges -- for instance, the N2 guanine atom is shared between the Sugar and Watson-Crick edges. ???

TODO tSS GG

Some base pairs defined by Westhof and Leontis do bind exclusively to the O2' ribose atom.
However, this is the exception to the rule, and it makes us ask whether we shouldn't call these pairs "nucleotide pairs" instead of "base pairs". -->


Example: cWW GC, tWW GC

The names are commonly shortened to 3-letter codes -- `c` or `t` for cis or trans, and `W`, `H` or `S` for each of the 2 edges.
`W` is Watson-Crick, `H` Hoogsten and `S` is the Sugar edge.
Since the pairing conformation depends on the pairing bases as well, we include the `A`, `T`, `U`, `G`, or `C` letters to identify the sequence.
For example, we can say that the image above shows the `cWW GC` and `tWW GC` pairs.

### M-N whatever and various adhoc terminology

TODO

### Comparison

The Leontis-Westhof system is more general, we can observe 122 of the LW base pairs in high quality X-ray structures deposited in the PDB.
The Saenger system is more restrictive, as it only includes the basepairs that have at least two hydrogen bonds between two polar base atoms.
Although one could argue that the other pair types do not form “real” basepairs, it is noteworthy that the Saenger system also overlooks certain base pairs that meet the criteria.
For instance, the **tWS CG** basepair which satisfies the condition is not included in the scheme.

![**tWS GC**: two polar base-to-base H-bonds, but not accounted for in the Saenger system](../img/tWS-CG-1jj2-9_46-9_4.png)

<!-- We have not seen a recent publication using the Saenger's scheme, but the neither the Leontis-Westhof system is universally adopted.
It is more general and maybe more importantly it is systematic -- we don't need to remember 28 numbers to be able to identify the basepair type when viewing a molecular structure. -->

<!-- Being more general, the L-W system includes pairs which some might not want to call "base pairs".
A number of described base pairs only bind with a single hydrogen bonds or requires binding to ribose O2', restricting the class to RNA.
However, a few of doubly bonded legitimate base pairs are missing in the Saenger system, for instance the XX. -->



