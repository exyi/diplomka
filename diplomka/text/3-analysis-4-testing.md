## Testing Basepair Parameters {#sec:testing-basepair-params}

To verify that the parameters proposed in the previous section [-@sec:basepair-params] are sufficient to define basepairs, we can try to replicate FR3D annotations.
We simply compute the parameters for all basepairs reported by FR3D and set the parameter boundaries to the observed minimum and maximum.
Then, we obtain all basepairs satisfying these constraints and examine the differences between this set and FR3D assignments.

This approach will in principle yield zero false negatives, but it is sensitive to extremes in FR3D annotations -- in the training set.
That is mitigated by using 
<!-- We have done a simple experiment to  — that it is  using them.
In the experiment, we simply try to replicate FR3D annotations by setting the boundaries at the lowest and highest observed value in basepairs reported by FR3D on the reference set.

We deliberately use a different set parameter than FR3D in order for this experiment to work.
Notably, we lack constraints on the relative translation of the pairing bases compared to FR3D.
Translational constraints are powerful, but because they are hard to generalize even across a single family, we can significantly simplify the model by avoiding them.
 -->
