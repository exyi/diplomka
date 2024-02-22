## Special cases of the Leontis-Westhof naming scheme

### Symmetry of L-W classes

As presented in the previous chapter, the L-W scheme theoretically allows 156 possible distinct base pair conformations, if we assume uracil is equal to thymine (TODO this is maybe interesting).
On each side of the pair, we have 3 options for the edge and 4 options for the nucleotide, 12 options in total.
We can create a $12 \times 12$ matrix of base-edge combinations.
We assume that the pairs are symmetric, for example that **cWW GC** is only flipped **cWW CG** and **cWH AG** is flipped **cHW GA**.
To get the number of symmetric combinations, we count the number of elements in the lower triangular matrix -- we ignore all entries above the diagonal.
This equals $12 + 11 + \cdots + 1 = 78$.
Each base-edge couple can pair in **cis** or in **trans**, multiplying by the additional 2 options, gives us 156 options.

It is not hard to see that not all "possible" pairs can make sense.
For example, the Hoogsteen edge of uracil is a single atom -- a doubly-bonded oxygen.
Thus, **cHH UU** simply isn't an energy favorable base interaction.

<!-- The conformation is different for each of the four nucleotides.
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

$$2 \cdot \left( 16 \cdot 3 + 10 \cdot 3 \right) = 156$$ -->

### Sugar-Sugar symmetry

Unfortunately, the assumption of symmetry we made above is generally invalid.
Many Sugar-Sugar basepairs were from the start defined asymetrically, **cSS AC** isn't a flipped **cSS CA**.
All six “symmetrical” pairs are defined differently in **cis** Sugar/Sugar, and only one in **trans** Sugar/Sugar (GA/AG). <!--TODO Ref-->

This property isn't explicitly mentioned in the LW2001 paper (TODO verify again), however the tables of examples in LSW2002 clearly show different base pairing for the two cases.

![The C-A pair is different from A-C, it is shifted by ~3 Å horizontally resulting in a different set of H-bonds](../img/cSS-CA-vs-AC.png)

It is definitely not a mistake.
Leontis with Westhof also propose a glyph symbols for each base pair family.
We can show Watson-Crick edge as a circle (○), Hoogsteen edge as square (□) and sugar edge as an triangle (▷).
If the basepair is **cis**, it is filled (●, ■, ▶), if **trans**, it is empty (○, □, ▷).
The basepair families are then ⎼○⎼ (or ○⎼○) for **tWW**, ●-▶ for **cWS** and ⎼▶⎼ **cSS**.
Note that only the sugar edge has an asymmetric glyph symbol.

![In trans Sugar/Sugar, C-A is defined while A-C is left undefined. By contrast, in W/W and H/H the C-A and A-C show the same pair (turned upside down)](../img/tSS-CA-vs-AC.png)

Detailed discussion of software tools is in a following section, but it is important to note here, that FR3D distinguishes the **cSS** cases by lowering the one of the `S` letters, thus **cSs AC** and **csS AC** are different basepair geometries.

<!-- ■⎼▶
□⎼▷
○⎼● -->

### Alternatives

TODO tWWa CC, cWWa GT.

![](../img/tWW-U-U-vs-tWWa-U-U.png)

### Bifurcated hydrogen bonds

There is a special category for pairs involving a hydrogen bond between three atoms.
Either one hydrogen lies between two acceptors, or a single donor nitrogen atoms binds both hydrogens.
Since the bifurcated H-bond needs two acceptors or two donors next to each other, it is only defined for the Watson-Crick edge.

While the 2002 paper presents this category and at least FR3D reports it, it is often not considered.
The category only contains 6 distinct base pairs, none of which bind with at least two hydrogen bonds.
In this work, we mostly skip the analysis of these Watson-Bifurcated basepairs.
The provided scripts do process them, but we will avoid them in the discussion for brevity.
