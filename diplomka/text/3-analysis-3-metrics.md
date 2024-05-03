## Basepair Metrics

The objective of this work is to find and compare decent measures of basepairing, so we shall finally specify what we are looking for.
The main qualities of the metrics are:

* Interpretability — can we easily tell what the number means, can we easily guess the number by looking at the 3D structure?
* Simplicity — the ideal metric is easy to define and calculate algorithmically.
* Stability — it should have low variance for good quality basepairs.
* Universality — it should have similar value or similar variance on different basepair types.
* Independence — it is advantageous, if the metric is empirically independent on the other metrics we want to use.

<!-- We need to define basepairs in such a way, that humans can easily understand the results of computer calculations. -->

It is important that humans can easily understand why the computer did or didn't assign a specific basepair, either for debugging the software or potentially for refining the molecular structures.
Thus, we want the metrics to be reasonably simple and interpretable.

In order to be useful, the metric must have a sharp enough distribution to help identifying the specific basepair class.
As an extreme example, we can easily rule out using the euler angles shown in @fig:euler-angles-bad-distribution, because the values may span the entire range of -180° ... +180°.

![TODO](TODO){#fig:euler-angles-bad-distribution}


### The Number of Parameters

If we assume that the molecules are not deformed, then we can describe the relative position of two bases by six numbers.
The standard basepair parameters (@sec:std-base-parameters) can be one such description — essentially, we only need three translation coordinates and three angles of relative rotation.
However, we may not be able to characterize all the basepair types using simple numeric ranges of this minimal set of parameters.

Additionally, we consider it more elegant to avoid “hard cuts” in the data distributions.
As prof. Zirbel said in one of our meetings, we would prefer to have gentle limits on many parameters, than few uncompromising cutoffs.
FR3D indeed performs very well regarding this — we didn't find a single unnatural looking line in any of our scatter plots and histograms.
Since each of gentle limits cuts out a small fraction of the potential basepairs, the exact value of the limits isn't as sensitive.
That makes it easier to set the limit and also allows us to share the same conditions across all classes of a given basepairing family.

<!-- Since we do not  -->
We would rather avoid more complex constraints than a set of one dimensional numeric ranges.
Generalizing the constraints into two or more dimensions is similar to inventing additional parameters by linearly combining the existing ones, except that the potential new parameter is easier to share across the ~120 basepair classes.

TODO trans + YPR demo image

### Hydrogen bond lengths and angles

A good starting point is simply measuring the distance between the atoms forming the hydrogen bonds.
Traditionally, we would measure a hydrogen bond between the hydrogen and the acceptor heavy atom.
Since the exact hydrogen positions are often unknown, we will instead only consider the distance between heavy atoms (oxygen, nitrogen, or carbon).
Despite the availability of many algorithms for completing PDB structures with the missing hydrogens, there are tricky cases where automatic the completion fails.
Specifically, some bases may hold a charge and thus have an addition hydrogen, or it may be in a tautomeric form where the hydrogens are elsewhere.
Although it isn't common, it is crucial in some basepair classes, and it is likely biologically relevant (TODO https://doi.org/10.1002/cphc.200900687 ?, TODO https://www.ncbi.nlm.nih.gov/pmc/articles/PMC97597/ ?). TODO cWH-A-G

In addition to the distance, we can simply determine an angle between the two heavy atoms and a third atom situated on each base.
Depending on if we select the third atom next to the acceptor or the donor, we get what we will call "Donor angle" or "Acceptor angle".  For consistency, we will always select the neighbor atom with higher number.

![A canonical cWW GC basepair (3cpw A1103:A1240). The distance between O2 and N2 is indicated, in addition to the donor and acceptor angles of the other four involved atoms. Note that the hydrogens are not considered in the calculation, they were added in PyMOL to easily recognize donors and acceptors.](../img/cWW-GC-length-and-covalent-angles.png){#fig:cWW-GC-length-and-covalent-angles}
<!-- fetch 3lz0
select pair, 3lz0 and (chain J and resi 21 or chain I and resi \-20) -->

Unfortunately, our problem isn't as simple as measuring few distances.
Even if set strict limits on them, we will still get many false positives.
As shown in figure -@fig:cWW-GC-length-and-covalent-angles, an ideal cWW GC pair should have h-bond lengths of about 2.9 Å and all angles at about 120°.
We must allow some slack, as no ideal pair exists in reality — a 0.5 Å and 20° tolerance is quite conservative.
Yet, we still find a number of false positives similiar to the one shown in @fig:cWW-GC-false-positive-hbond-lengthsangles.
Toughening the limits slightly would dismiss this case, but we are already dropping many good examples, as anyone can try out in the
(basepairs.datmos.org)[https://basepairs.datmos.org/#cWW-G-C/hb0_L=..3.4&hb0_DA=100..140&hb0_AA=100..140&hb1_L=..3.4&hb1_DA=100..140&hb1_AA=100..140&hb2_L=..3.4&hb2_DA=100..140&hb2_AA=100..140&baseline_ds=fr3d-f] web application.


![A false positive find (3Lz0 J21:I-20) in the cWW GC class using solely the basic H-bond parameters. We can see that the distances and angles are adequate, but the bases are shifted by almost one step.](../../cWW-GC-false-positive-hbond-lengthsangles.png)
<!-- fetch 3lz0
select pair, 3lz0 and (chain J and resi 21 or chain I and resi \-20) -->

### Hydrogen bond planarity

After hydrogen bonds, the second most important feature of pairing bases is their coplanarity.
Coplanarity is not easily defined a single measure, but it essentially means that the planes of the two bases are not overly different.
One of our proposals is to measure how far do the hydrogen bonds deviate from the plane.
In the example shown in @fig:cWW-GC-false-positive-hbond-lengthsangles, the bonds are almost perpendicular to both of the planes.

Acceptor/Donor vs 1/2
