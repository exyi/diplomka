## Basepair Metrics

The objective of this work is to find and compare decent measures of basepairing, so we shall finally specify what we are looking for.
The main qualities of the metrics are:

* Interpretability - can we easily tell what the number means, can we easily guess the number by looking at the 3D structure?
* Stability - it should have low variance for good quality basepairs.
* Universality - can the same or similar measure be reused for other basepair types?
* Simplicity - the ideal metric is easy to define and calculate algorithmically.
* Independence - it is an advantage, if the metric is empirically independent on other metrics we want to use.

<!-- We need to define basepairs in such a way, that humans can easily understand the results of computer calculations. -->

It is important that humans can easily understand why a computer cannot Interpretability is important for 


### The Number of Parameters

The relative position of two bases can be completely described by six numbers — the standard basepair parameters (@sec:std-base-parameters) are one such description.
However, using such parameters may not lead to the simplest description of all basepair types.
First, aiming for simplicity, we would rather avoid more complex constraints than one dimensional numeric ranges.
<!-- Generalizing the constraints into two or more dimensions is similar to inventing additional parameters by linearly combining the existing parameters, except that the new parameter is easier to share across the ~120 basepair types. -->
Second, having different constraints on all basepair types is about a hundred-times more rules than having a shared 


TODO demo image
