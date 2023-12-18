## Special cases of the Leontis-Westhof naming scheme

### Symmetry of L-W classes

The L-W scheme theoretically allows 156 possible distinct base pair conformations.
The conformation is different for each of the four nucleotides, we assume that uracil is the same as thymine.
We have the following choices:

* **cis** or **trans**
* **pair** of **nucleotides**, choice of 2 out of 4
* **pair** of **edges**, choice of 2 out of 4

Since both choices allow repetitions (`A` can pair with another `A`), we calculate the number of options as $\binom{n + 1}{2}$ (https://en.wikipedia.org/wiki/Combination#Number_of_combinations_with_repetition).
Since the numbers are small, we can alternatively count the number of options using a simple table.

However, we have to account for the fact that some pairs are symmetric to each other and avoid double counting these.
For example, the `cWW GC` is the same pair as `cWW CG`.
Generally, we should avoid counting a pair if swapping the order of edges and the order of nucleotides yields a pair which was counted already.
We will first avoid edge combinations marked as duplicate:

| - | W | H | S |
|---|---|---|---|
| W | 1 | Dup | Dup |
| H | 2 |  3  | Dup |
| S | 4 |  5  |  6  |

This leaves with 6 edge combinations, 3 symmetric and 3 asymmetric.
If the edge combination is asymmetric, we can count all 16 nucleotide combinations.
If is it symmetric, we can only count the 10 unique nucleotide combinations:

| - | A | T   | G   | C   |
|---|---|---|---|--|
| A | 1 | Dup | Dup | Dup |
| T | 2 |  3  | Dup | Dup |
| G | 4 |  5  |  6  | Dup |
| C | 7 |  8  |  9  | 10  |

When we add these and multiply by 2 to account for the **cis** or **trans** choice, we get:

$$2 \cdot \left( 16 \cdot 3 + 10 \cdot 3 \right) = 156$$

### False symmetry

Unfortunately, the symmetry equivalence explained in the previous section, does not always hold.
Example: cSS CA vs cSS AC

### Alternatives

There are multiple ways blabla just look at tWWa CC, cWWa GT.

### Bifurcated hydrogen bonds

A special category was created for pairs involving a hydrogen bond between three atoms.
Either one hydrogen lies between two acceptors, or a single donor nitrogen atoms binds both hydrogens.
Since two acceptors or two donors next to each other are required, this is only defined for the Watson-Crick edge.

While this category is presented in the 2002 paper and reported at least by the FR3D program, it is often not considered.
The category only contains 6 distinct base pairs, none of which bind with at least two hydrogen bonds.

In this work, we mostly skip the analysis of these pase pairs.
The provided scripts allow processing them, but we avoid them in the written text for brevity.
