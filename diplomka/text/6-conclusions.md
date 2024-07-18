# Conclusions {#sec:concl}

In this thesis, we defined a set of clear and unambiguous geometric parameters (@sec:basepair-params) that discriminate between basepair classes as defined in the Leontis-Westhof system.
These parameters can therefore be used for basepair assignment in molecular structures, as we have verified in [section @sec:testing-basepair-params] both on the References set of quality-checked structures (@sec:filter) as well as in all nucleic acid containing structures in the whole PDB archive.

The exact basepair class definitions using these parameters are in the final phase of testing.
To facilitate the testing and tuning, we developed a web-based application (@sec:tuning-app) that allows iterative manipulation of the upper and lower limits of all parameters.
The basepair reassignment is visualized in real time making the process interactive.
The application is publicly available at [basepairs.datmos.org](https://basepairs.datmos.org){.link-no-footnote} and may also be used to browse the diversity of non-canonical basepairs or lookup structures with interesting pairs.

After the initial batch processing of PDB structures, it is a purely client-side web application hosted as a set of static files --
making it easily executable on users' computers and also making it easier to keep operational in the future (@sec:impl-webapp-sustainability).

## Ongoing work {#sec:concl-future}

The working group currently work on finishing the basepair class definitions for the process of basepair assignment and prepares a manuscript describing the assignment procedure.
We are also expanding the process to include an approximate assessment of each basepair quality.

As the next step, we plan to integrate the new basepair assignment into the [DNATCO web server](https://dnatco.datmos.org), designed to [explore and validate individual structures](https://doi.org/10.1107/S2059798318000050).
DNATCO primarily focuses on the nucleic acid backbone structure and merging that with the understanding of basepair interactions will hopefully lead to new insights into nucleic acid structural biology.

<!-- GPT4:

In this thesis, we have made significant strides toward providing clear and unambiguous definitions for the basepair classes of the Leontis-Westhof system. The lack of accessible and definitive selection criteria in existing software has posed a considerable challenge for structural biologists. Our work addresses this gap by establishing a comprehensive framework that includes the definition and implementation of measurable parameters critical for basepair classification.

One of the notable achievements of our project is the development of an interactive application that allows for the precise tuning of selection criteria for each basepairing class. This tool not only aids in the accurate annotation of basepairs but also serves as a valuable resource for the scientific community, facilitating further research and exploration in nucleic acid structural biology.

Additionally, we have made significant progress in the integration of basepair information into the DNATCO web server. While DNATCO primarily focuses on the nucleic acid backbone structure, our integration of basepairing information aims to provide a more holistic understanding of nucleic acid structures by combining backbone and base interaction data.

The filtration of structural data to create the Reference Set was another pivotal aspect of our work. The methodological rigor applied ensures a high level of data quality, thereby minimizing bias and enhancing the reliability of our results. This curated dataset will be invaluable for future analyses and the continued development of our basepair classification framework.

Looking ahead, our team is already working on expanding the basepair definitions and integrating them into the broader understanding of nucleic acid structures. Future endeavors include the continued refinement of our methodologies and the exploration of new insights that arise from combining backbone and base interaction data.

In conclusion, this work lays a strong foundation for advancing the field of nucleic acid structural biology. By providing clear definitions, robust tools, and high-quality data, we aim to support ongoing research and foster new discoveries in this vital area of study.
-->
