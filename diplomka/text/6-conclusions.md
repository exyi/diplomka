# Conclusions

We defined a set of geometric parameters (@sec:basepair-params) that discriminate between basepair classes as defined in the Leontis-Westhof system.
Application of these parameters can therefore be basepair assignment in molecular structures, as we have verified in [section @sec:testing-basepair-params] both on the References set of quality-checked structures (@sec:filter) as well as in all nucleic acid containing structures in the whole PDB archive

TODO ehhh The parameter limits assigning the optimal sets of base pairs (...on RS+PDB?) are in final phases of their testing.
To facilitate the testing and tuning, we developed a web-based application (@sec:tuning-app) that allows iterative manipulation of the upper and lower limits of all parameters.
The application is running at [basepairs.datmos.org](https://basepairs.datmos.org){.link-no-footnote}.
The basepair reassignment is visualized in real time making the process interactive.

After the initial batch processing, it is a purely client-side web application hosted as a set of static files --
making it easily executable on users' computers and also making it easier to keep operational (@sec:impl-webapp-sustainability).

## Ongoing work

Our team currently works on finishing the basepair class definitions for the process of basepair assignment and prepares a manuscript describing the assignment process.

As the next step, we plan to integrate the new basepair assignment
into the [DNATCO web server](https://dnatco.datmos.org),
designed to [explore and validate individual structures](https://doi.org/10.1107/S2059798318000050).
Merging the understanding of nucleic acid backbone structure and the basepair interactions will hopefully lead to new insights into the nucleic acid structural biology.
