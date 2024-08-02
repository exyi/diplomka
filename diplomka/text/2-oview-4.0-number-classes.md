## Special Cases of the Leontis-Westhof Naming Scheme {#sec:bp-terminology-lw-edgecase}

### Symmetry of L-W classes

As presented in the previous section, the L-W system theoretically distinguishes 156 distinct classes of basepairs, when treating uracil as equivalent to thymine.
On each side of the pair, there are three options for the edge and four options for the nucleotide, resulting in a total of twelve unique combinations -- the L-W families.

We can create a $12 \times 12$ matrix of base-edge combinations.
Assuming symmetry between pairs, such as **cWH A-G** being equivalent to its flipped counterpart **cHW G-A**, we count the number of elements in the lower triangular matrix, ignoring all entries above the diagonal.
This results in $12 + 11 + \cdots + 1 = 78$ unique combinations.
Given that each base-edge couple can pair in either **cis** or in **trans** conformation, we multiply this by an additional two options, resulting in a total of **156** possibilities.

It is evident that not all theoretically possible pairs can make sense chemically.
For instance, the Hoogsteen edge of cytosine consists of a single NH<sub>2</sub> group, which can only act as a hydrogen bond donor.
Thus, a **cHH C-C** basepair, involving only this edge, cannot provide a stabilizing base interaction.

<!-- The conformation is different for each of the four nucleotides.
We have the following choices:

* **cis** or **trans**
* **pair** of **nucleotides**, choice of 2 out of 4
* **pair** of **edges**, choice of 2 out of 4

Since both choices allow repetitions (`A` can pair with another `A`), we calculate the number of options as $\binom{n + 1}{2}$ (https://en.wikipedia.org/wiki/Combination#Number_of_combinations_with_repetition).
Since the numbers are small, we can alternatively count the number of options using a simple table.

However, we have to account for the fact that some pairs are symmetric to each other and avoid double counting these.
For example, the **cWW G-C** is the same pair as **cWW C-G**.
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

![The C-A pair is different from A-C -- it is shifted by ~3 Å horizontally, resulting in a different set of H-bonds <https://doi.org/10.1093/nar/gkf481>](../img/cSS-CA-vs-AC.png){#fig:cSS-CA-vs-AC}

Unfortunately, the assumption of symmetry we made earlier is generally invalid, as Sugar/Sugar pairs are actually defined asymmetrically — for instance, **cSS A-C** is not a flipped version of **cSS C-A** (@fig:cSS-CA-vs-AC).
[Leontis and Westhof](https://doi.org/10.1017/s1355838201002515) only define **cis** Sugar/Sugar as non-symmetric, while **trans** Sugar/Sugar should be symmetric:

> The cis and trans W.C./W.C., the trans Hoogsteen/Hoogsteen, and the trans Sugar-edge/Sugar-edge geometries (…) are symmetric, with the interacting bases related by a twofold rotation about an axis passing either vertically or horizontally through the center of the basepair.
> **The cis Sugar-edge/Sugar-edge geometry (…), however, is not symmetric.**
> To illustrate this point, two different A-G cis Sugar-edge/Sugar-edge pairs are shown in Figure 7. [note: we show A-C in @fig:cSS-CA-vs-AC]

In spite of that, the tables of examples in [their 2002 publication](https://doi.org/10.1093/nar/gkf481) do not treat Sugar/Sugar symmetrically either -- see @fig:tSS-CA-vs-AC.
However, only the **tSS A-G** and **tSS G-A** classes are defined in such _conflict_, and they differ only slightly.

![In trans Sugar/Sugar, C-A is defined while A-C is left undefined. In contrast, in W/W and H/H the C-A and A-C show the same pair (turned upside down)](../img/tSS-CA-vs-AC.png){#fig:tSS-CA-vs-AC}

Leontis and Westhof also propose a glyph symbol for each basepair family:
the Watson-Crick edge as a circle (`○`), Hoogsteen edge as square (`□`) and the sugar edge as a triangle (`▷`).
If the basepair is **cis**, it is represented with filled symbols (`●`, `■`, `▶`); if **trans**, it is shown with empty symbols (`○`, `□`, `▷`).
The resulting basepair families are then represented as `⎼○⎼` (or `○⎼○`) for **tWW**, `●⎼▶` for **cWS**, and `⎼▶⎼` for **cSS**.
Note that only the sugar edge has an asymmetric symbol, intuitively indicating its asymmetry and allowing placing of the symbol on a diagram in any orientation.

#### FR3D family lowercasing {#sec:bp-fr3d-lowercasing}

[Section @sec:software] discusses available software tools more thoroughly; however, it is worth noting here how the FR3D program disambiguates the two **cSS** cases by lowering the second **S** letter.
For easier lookup by nucleotide number, FR3D reports all basepairs in both orientations, e.g., **cWW G-C** is also reported as **cWW C-G**.
If this would lead to ambiguities due to asymmetry, the second edge letter is lowered.
For instance, a **cSs A-C** pair is also reported as the corresponding **csS C-A** pair, both meaning **cSS A-C** in the Leontis-Westhof terminology.
FR3D does not lower the second **S** if the other orientation is left undefined.

FR3D also lowers the second edge letter if the full class name is symmetric, but the defined H-bonds are not identical when flipped.
For instance, the H-bonds in **cWW A-A** pairs would be ambiguous, as these pairs bind from **N6** to **N1**, and from **N1** to **C2**.

<!-- ■⎼▶
□⎼▷
○⎼● -->

### Alternative H-bond sets {#sec:bp-terminology-lw-edgecase-a}

In a few classes, it is possible that the bases can interact by two possible sets of H-bonds on the same edges and the same cis/trans orientation.
We observe these _subclasses_ for in **cWW G-U**, **tWW C-C** and **tWW U-U** (@fig:tWW-U-U-vs-tWWa-U-U), but it is possible that other, less common ones, exist.
The **cWW G-U** pair is especially interesting because
[one of the subclasses is anionic. One of the bases is charged by losing a hydrogen atom leading to an atypical donor position.](https://doi.org/10.1261/rna.079583.123)

In these cases, FR3D appends an **“a”** to the family name to distinguish subclasses.
For instance, the standard **cWW G-U** is labeled as **cWW**, and the anionic form is labeled **cWWa**.
We are not aware of other programs capable of classifying subclasses.
In this work, we follow the FR3D convention, although we have had debates on revisiting subclass naming in the working group.

<!-- tWWa CC, cWWa GT. -->

![The two alternatives of the **tWW U-U** pair. Although the only Watson-Crick edge is involved in both cases, two pairs of hydrogen bonds are possible and both options are well populated in PDB structures.](../img/tWW-U-U-vs-tWWa-U-U.png){#fig:tWW-U-U-vs-tWWa-U-U}

<!-- ### Bifurcated hydrogen bonds

Finally, to underline the saying that [in biology, there are a thousand
exceptions to each rule](https://tandy.cs.illinois.edu/Hunter_MolecularBiology.pdf), a special category for basepairs between the three edges also exists.
In these pairs, two hydrogen bonds are formed onto the single _corner_ atom.
Specifically, two donors may share a single acceptor, or a NH<sub>2</sub> group can have both hydrogens bound.
Since bifurcated H-bonds require two adjacent acceptors or donors, they are only defined for the Watson-Crick edge.

Our pipeline does process these pairs, there is nothing special about them, but we will not discuss this category in the following text further. -->

<!-- However, we will not discuss this category further here, as we are already getting lost in edge cases.
 -->

<!-- 
While the 2002 paper presents this category and at least FR3D reports it, it is often not considered.
The category only contains 6 distinct basepairs, none of which bind with at least two hydrogen bonds.
In this work, we mostly skip the analysis of these Watson-Bifurcated basepairs.
The provided scripts do process them, but we will avoid them in the discussion for brevity.

Mixtral: The text is clear and mostly well-written. However, there are some minor stylistic improvements that can be made for consistency and readability. Consider revising to: "The 2002 paper introduces this category, which FR3D also reports; however, it is not commonly considered in the literature. This category comprises only six distinct basepairs, none of which engage in at least two hydrogen bonds. In our work, we generally omit an analysis of Watson-Bifurcated basepairs for brevity. The provided scripts do process these cases; however, we will intentionally exclude them from the discussion."
-->
