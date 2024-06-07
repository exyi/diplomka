# Software Implementation

This section describes the technical details of the pipeline projected in the previous chapter.
The aim is to facilitate reproducibility of the results, explain some technical decisions, as well as provide a place to acknowledge and cite other software this project relies on.
Non-technical readers can safely skip the entire chapter, we separated the technicalities from the theory precisely for that reason.


![Overall flow of data through the pipeline. First, we run FR3D or another source of a list of basepairing nucleotides. Second, `pairs.py` is used to calculate the parameters for all basepairs. Then, we use `pair_distributions.py` to perform global analysis of each class of basepairs, and `gen_contact_images.py` to render molecular images for the web application.](../img/diagram-overall-dataflow.svg)
