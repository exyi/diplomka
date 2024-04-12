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
As an extreme example, we can easily rule out using the euler angles shown in @fig:euler-angles-bad-distribution.svg, because the values may span the entire range of -180° ... +180°.

![TODO](TODO){#fig:euler-angles-bad-distribution.svg}



### The Number of Parameters

The relative position of two bases can be completely described by six numbers — the standard basepair parameters (@sec:std-base-parameters) are one such description.
However, using such parameters may not lead to the simplest description of all basepair types.
First, aiming for simplicity, we would rather avoid more complex constraints than one dimensional numeric ranges.
<!-- Generalizing the constraints into two or more dimensions is similar to inventing additional parameters by linearly combining the existing parameters, except that the new parameter is easier to share across the ~120 basepair types. -->
Second, having different constraints on all basepair types is about a hundred-times more rules than having a shared 

The relative position of two bases can be completely described by six numbers — three for translation of a reference frame, three for their relative rotation (using Euler angles).
However, such description is not necessarily the best for constraining 

TODO demo image


### HB/Plane

Acceptor/Donor vs 1/2
