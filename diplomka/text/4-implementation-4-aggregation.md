## Data Aggregation Scripts — `pair_distributions.py`

The `pair_distributions.py` has multiple, but relatively simple jobs.
First, it calculates the KDE densities for a given table of all pairs.
It then generates a PDF file with histograms or KDE densitygrams of selected parameters for each class.
Since we always wanted a different set of parameters in a different layout, the parameter selection is not exposed as a CLI option and is instead configured in the source code.
The attached version of the script includes histograms for almost all parameters.
Additionally, it can generate overview 2D KDE pairplots, but the code is commented out in the attachment due to its heavy impact on runtime.
To quickly introduce each basepair class, the script finds the basepair closest to ideal (@sec:opt-params-exemplar) and renders its image (@sec:impl-basepair-img).

We also need to run this script to prepare the Parquet files for the web application, as it completes the tables with the `mode_deviations` and `kde_likelihood` columns.
This feature must be enabled with the `--reexport=partitioned` option, as calculating the KDE likelihood for each datapoint is rather time intensive and would unnecessarily slow down rendering of the histograms.

Lastly, the output folder will also contain a CSV file with overall statistics for each class, and a separate table of parameter boundaries, ready for use in @sec:testing-basepair-params.
