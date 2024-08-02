
# Software implementation {#sec:impl}
This section outlines how we have implemented the data processing and presentation pipeline.
The aim is to facilitate reproducibility of the results, explain some technical decisions, as well as provide a place to acknowledge and cite other software this project relies on.
Non-technical readers can safely skip the entire chapter; we separated the technicalities from the theory precisely for that reason.


![Overall flow of data through the pipeline. First, we run FR3D or another source of a list of basepairing nucleotides. Second, the `pairs.py` script is used to calculate the parameters for all basepairs. Then, we use the `pair_distributions.py` script to perform global analysis of each class of basepairs, and `gen_contact_images.py` to render molecular images for the web application. Optionally, we can include parameters calculated by DSSR.](../img/diagram-overall-dataflow.svg)
