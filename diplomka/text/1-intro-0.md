# Introduction

<!-- Since the [initial discovery of DNA structure](https://doi.org/10.1038/171737a0),  -->

TODO zapracovat Bohdanovo review

Nucleic acids are some of the most important biological molecules, and the study of their structure and function is critical to understanding the fundamental principles of life.
The [well known double-stranded helix](https://doi.org/10.1038/171737a0) dominates the structure of DNA, protecting its valuable information content.
On the other hand, RNA molecules often form complicated structures, often with dynamic catalytic domains such as those in the ribosome <https://doi.org/10.1201/9781315735368> TODO cit The_Ribosome_Is_a_Ribozyme page 346.
A number of RNAs can even have catalytic activity independent on proteins (<https://doi.org/10.1261/rna.031401.111>, <https://doi.org/10.1038/nchembio.1846>).
Similarly to the DNA helices, these complex structures are also driven by the pairwise interaction of nucleotide bases.
However, these interactions have more diverse forms than the two most common basepairs described by Watson and Crick.

Given the diversity of non-canonical basepairs, it is unsurprising that they require more research than their canonical counterparts.
A significant step towards addressing the gap is the [Leontis-Westhof classification system](https://doi.org/10.1017/s1355838201002515), introduced in 2001, which provides a framework for categorizing and naming non-canonical basepairs.
A [subsequent publication](https://doi.org/10.1093/nar/gkf481) shows more than a hundred different basepairing conformations in experimental structures archived in the [PDB database](https://doi.org/10.1093/nar/gky949).
Today, with the availability of about 17 000 nucleic acid structures, we are in a better position to properly describe even the rarer basepair kinds.

If we analyze the entire PDB database with one of the available basepair assignment programs (FR3D, @sec:software-FR3D), we can see that Watson-Crick pairs dominate all other basepair classes.
However, in RNA the total cardinality of all non-canonical pairs is actually larger than the canonical ones; meaning that canonical pairs represent slightly below 50% of the total.
Even in DNA structures, 25% of all basepairs are non-canonical, notable biologically relevant structures are the [i-Motif](https://doi.org/10.1093/nar/gky735) and the [G-quadruplex](https://doi.org/10.1016/j.trechm.2019.07.002).
<!-- However, it must be noted that the PDB is heavily biased towards _interesting_ structures -->

## Aims of the Project


The higher-level objective of our working group is to provide unambiguous definitions for the basepair classes of the Leontis-Westhof system.
At present, a number of rigorous definitions exists, as each computer program capable to annotating basepairs requires one.
However, with the exception of few overly simplistic algorithms, the exact selection criteria are not published outside the source codes, rendering it largely inaccessible to a significant portion of structural biologists.

In this work, we establish a framework necessary for achieving the objective.
We define and implement the measurable parameters which will be subsequently utilized to define the basepairs (@sec:basepair-params).
Additionally, we develop an interactive application that enables the tuning of selection criteria for each basepairing class (@sec:tuning-app).

![Our plan to improve the shortcomings of existing programs for basepair assignment. XKCD #927 by Randall Munroe](../img/xkcd_standards_2x.png){#fig:xkcd_standards_2x}


<!-- Even though the number of solved nucleic acid structures is significantly lower than the number of known protein structures,  -->

<!-- | Basepair | DNA | RNA |
|----------|-----|-----|
| Watson-Crick A-T -->


<!-- canonical Watson-Crick G-C and A-T basepairs only account for about one half of the interactions in the RNAs we studied.

* DNA is usually the classic helix described by Watson, Crick, and Franklin back in 1953
* RNA often forms much more complex structures, but also contains the helixes
* RNA helix is usualy A-form, DNA B-form, but the basepairs are practically the same
* 
... At the time of writing we have more than 7 thousand RNA structures in PDB, and almost 2 thousand of them are pure nucleic acid, without any protein
... 

"Only" 61.4% of all found basepairs are canonical Watson-Crick (in all PDB structures < 3.5 Å) -->

<!--
Promptses

Write me about 1-2 pages introduction to a thesis about non-canonical basepairs. Include the following information: "Watson-Crick basepairs form double helix, which is the majority of basepair, but 'only' ~75% in DNA and ~50%. DNA has biologically relevant i-Motif (with charged cytosine pairs) and guanine quadruplexes. RNA has huge amount of interesting structures, it even capable of catalytic activity independent of proteins. A simple basepair example is the wobble pair, crucial in tRNAs, but also important elsewhere. The number of NA structures in PDB is small compared to proteins, but grows just as quickly and is more than sufficient for identification and classification of these basepairs. The Leontis-Westhof classification system is from 2002, when there was maybe 1800 structures, but today we have more like 17k. A number of automated tools exist to annotate basepairs in 3D structures, with various shortcomings as described in @sec:3.2. Apart from analysing all types of basepairs and providing update for the definition images from [LSW2002], we propose a few new ways to measure basepair parameters, which can be used to define them and automatically assign them."


Mixtral:

The traditional Watson-Crick basepairs, which form the iconic double helix structure, are the most common basepairs in both DNA and RNA. However, they only account for approximately 75% of basepairs in DNA and 50% in RNA. This leaves a significant portion of nucleic acid (NA) structures unaccounted for, which are classified as non-canonical basepairs. These non-canonical basepairs play a crucial role in various biological processes, including gene regulation, protein synthesis, and chromosomal crossover.

Despite their importance, non-canonical basepairs have been historically understudied due to the lack of appropriate tools and classification systems. The Leontis-Westhof classification system, introduced in 2002, was a significant step towards addressing this gap, providing a framework for categorizing and analyzing non-canonical basepairs. However, with the rapid growth of NA structure data, there is a need for an updated and more comprehensive classification system.

Today, with the availability of more than 17,000 NA structures, there is a wealth of data that can be used to identify and classify non-canonical basepairs. Automated tools have been developed to annotate these basepairs in 3D structures, although they have certain limitations, as discussed in section 3.2.

In this study, we propose a comprehensive analysis of all classes of non-canonical basepairs, including an update of the definition images from the Leontis-Westhof classification system. Additionally, we aim to develop a more robust and comprehensive classification system that can accurately categorize and analyze non-canonical basepairs in NA structures. This work will contribute to a better understanding of the role of non-canonical basepairs in biological processes and provide valuable insights into their functional significance.

In summary, non-canonical basepairs are crucial components of NA structures and play a vital role in various biological processes. With the availability of a large number of NA structures and advanced computational tools, it is now possible to conduct a comprehensive analysis of non-canonical basepairs and develop a more accurate and comprehensive classification system. This work will provide valuable insights into the functional significance of non-canonical basepairs and contribute to a better understanding of NA structure and function.
-->
