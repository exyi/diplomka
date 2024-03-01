## Optimal parameter values

<!-- We totally understand that even if two basepairs are in the same category, they shouldn't be necessarily equal.
In the same conditions, maybe yes, there should be an optimal geometry with minimal energy.
But biological structures are interesting because they aren't the same, and thus the basepairs are not in the same conditions.
Actually, one of the main goals of this project is to show that a basepair type is a distribution, not just a single idealized example as was shown in LSW2002.
We wouldn't find many interactions in real data, if we insisted on the optimal conformation.
Most of the measured data is however noise, remember we dealing with 3Å resolution.
So it is useful to extract the mean value from it, if only to then find out how far does a given example deviate from it. -->

Given the inherent variability of biological structures, we have to recognize that a basepair category cannot be simplified into a single optimal value for each measured parameter.
Under identical conditions, a single geometry with minimal energy most likely exists, but basepairs in biological structures are seldom in the same conditions.
This project aims to demonstrate that a basepair type is better represented as a distribution, rather than a singular idealized example.
By insisting on the optimal conformation, we would either observe nucleic acids only rarely interact, or we'd have to classify most PDB structures are erroneous.

However, we also have to acknowledge that the measured parameters are inherently noisy, considering the 3 Å resolution at which we are working.
Therefore, extracting the mean value from the data is useful, if only to determine the degree of deviation of a specific example.

<!-- Even though basepairs are in the same category, 
Our goal is to show that each of the measured parameters has a distribution -->

<!-- We will not bother ourselves with the canonical Watson-Crick A-U and G-C pairs too much, since they are well studied already. But this is great for calibrating our measurements to the well-known optimal canonical basepair parameters. -->
In this work, we will not focus extensively on the canonical Watson-Crick A-U and G-C base pairs, given that these pairs are already extensively studied.
However, these base pairs will prove very useful in calibrating our measurements to the well known optimal canonical basepair parameters.
We'll use the [hydrogen-bond lengths provided as restraints for building nucleic acid structures in](https://doi.org/10.1107/S2059798321007610) as the golden values for the comparison.

### Choice of a midpoint statistic

We have a number of options how to calculate the middle value of a distribution.
Obvious choices are an arithmetic mean and a median — a 50th percentile.
We might however use any other percentile, maybe 45th percentile could have better properties for our use case.
Calculating a mean of a measured hydrogen length might be prone to outliers on the high end, so we could choose to filter out 10 percentiles from each end before the calculation.

#### KDE Mode

A more advanced option is to utilize a [Kernel Density Estimate](https://en.wikipedia.org/wiki/Kernel_density_estimation) of the distribution and measure its mode, i.e., find maximum of the KDE function.
The density estimate requires a bandwidth parameter which has a large impact on the resulting distribution.
The bandwidth is conceptually similar to a bin width in a histogram, the KDE is similar to a histogram made of Gaussian curves instead of rectangles.

A few methods of automatic bandwidth determination exist, we'll use the Scott's formula recommended in `scipy` documentation [TODO cite].
The automatic bandwidth depends on the number of samples and on the variance in the dataset.
We can also specify an adjustment of the computed value -- for instance, we could make it 1.5-times larger than Scott's formula thinks is the optimum.

<!-- Since we might want to compare several programs and experiment with changing the threshold, the resilience to outliers of the KDE mode is very useful.
It is also advantageous that it will select the highest peak in case of multi-modal distribution.
However, we have to verify that it is sufficiently stable and accurate on smaller samples.
It is also not great that it is computationally significantly more intensive, the implementation using `scipy` a has quadratic algorithmic complex.
Plus we cannot get a confidence interval by simply considering the standard deviation like to mean, we'll have to do bootstrap. -->

The KDE mode's resilience to outliers and multi-modal distributions make it useful for comparing various programs and manipulating threshold settings.
However, it is crucial to verify its stability and accuracy when dealing with smaller sample sizes.
Furthermore, it has a significantly higher computational cost than a simple mean or median, the implementation using `scipy` a has quadratic algorithmic complexity.
Additionally, the determination of a confidence interval requires bootstrapping or resampling, instead of the simpler application of central limit theorem.

#### Comparing the methods

We can make a simple experiment to choose a midpoint statistic seems to perform the best.
The criteria will be:

1. Accuracy -- how close to the reference H-bond lengths we are for canonical **G-C** and **A-U** pairs
2. Stability on smaller samples -- we'll compare the variance of the statistic itself on other samples.
3. Stability on skewed samples -- we'll cut out part of the dataset based on one parameter and observe how much do other parameters change.

We will the mean absolute difference between the statistics on small samples with the reference hydrogen bond lengths of canonical basepairs.
One hundred is a typical number of cases of a non-canonical basepair type, so we'll use sample size of 100.
Many classes have much fewer found cases, but we don't want to optimize the method for cases where we won't be able to produce a reliable number anyway.
We have over 30 thousand datapoints in canonical classes, so 1000 sampled batches should properly cover the available dataset.

The following figure compares various bandwidth adjustments of the default Scott's factor.
We can see that different hydrogen bonds have different optima, but it is clear that the general optimum is in the range from 1.0 to 1.5.


![[Reference H-bond lengths](https://doi.org/10.1107/S2059798321007610) / KDE mode with bandwidth adjustment 0.5 … 3.0](../img/KDE_bandwidth_golden_length_deviation.svg)

In the next figure, we can see that percentiles are almost as accurate as the KDE mode.
Interestingly, 46th percentile is the closest to the reference values on average.
However, as shown in the right plot, the optimal value varies widely across the different hydrogen bonds.
This clearly shows that observations do not exactly correspond with the reference values -- we are usually measuring shorter length of **GC N2 · · · O2** than the reference, so the 60th percentile is closest.
The variance in the KDE bandwidth can be explained similarly, it is only not as obvious which way the parameter biases the result.

![[Reference H-bond lengths](https://doi.org/10.1107/S2059798321007610) / percentile 30 … 70](../img/percentile_golden_length_deviation.svg)

The second criterion is the stability on the small samples.
We'll measure the statistic across the entire dataset and compare its mean absolute deviation with that of 1000 samples of size 100.In this case, we don't need to restrict ourselves to the H-bond lengths, so we will use all parameters we are currently working with (see the following sections).

Since not all parameters are in the same range, we cannot compute the mean error as we did for the H-bond lengths.
Instead, for each parameter, we will divide each deviation by the average deviation for this parameter.
Say, we have a matrix $D \in \mathbb{R}^{n \cdot m}$ of deviations for each parameter and each statistic.
Then the relative deviations are matrix $R$ of the same size:


$$R_{ps} = \frac{D_{ps}}{m^{-1} \cdot \sum_{k} D_{pk}}$$

The following figure shows, that the percentile is significantly more stable when compared to the KDE mode.

TODO compare effect size with ^


![Stability of the mode and percentile on various parameters of canonical GC and AU pairs. The error bars show the variance for different measured parameters.](../img/percentile_kdemode_stability_GC_AU.svg)


TODO: The 3., biased distributions

#### Conclusion

The median will serve as the primary point of reference in the analysis.
However, if the KDE mode deviates noticeably from the median, it may signify a more intricate distribution, necessitating further exploration.
It may be a bimodal distribution or one with a long tail on one side.
The KDE will have a bandwidth factor of 1.5 (TODO actually do so).
Both median and the KDE mode will thus be included in our output tables.
