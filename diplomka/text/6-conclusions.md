# Conclusions

We have successful defined and implemented a comprehensive set of parameters describing for basepairs (@sec:basepair-params).
These parameters have been demonstrated to work as a method for basepair assignment (@sec:testing-basepair-params), although it requires further tuning of the selection criteria to be accepted by the structural biology community.

To assist in the manual work of parameter tuning, we have implemented a tool for interactive visualization of the parameter distributions and for comparing the selected sets of basepairs (@sec:tuning-app).
After the initial batch processing, it is a web-based application hosted as a set of static files, with the computation happening client-side -- making it easily executable on users' computers and also unlikely to be a maintenance burden on the web administrator.
The application is running at https://basepairs.datmos.org, 


### Future work

Our team is already busy expanding the work in @sec:testing-basepair-params, and defining all basepair classes using the proposed new parameters.
This will allow us to create new method for basepair assignment.

The web-based basepair browser primarily for exploring various types of basepairs in the entire database.
We also plan to integrate the basepairing information into the [DNATCO web server](https://dnatco.datmos.org), designed to [explore and validate individual structures](https://doi.org/10.1107/S2059798318000050).
DNATCO currently focuses on the nucleic acid backbone structure, disregarding the importance of base interactions.
This work, on the other hand, ignores the sugar and phosphate conformation, but merging the understanding of both will hopefully lead to new insights into the nucleic acid structural biology.
