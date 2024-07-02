## Interactive Application for Parameter Tuning {#sec:tuning-app}

In the preceding section [-@sec:testing-basepair-params], we have seen how each class of basepairs behaves somewhat differently.
Moreover, it is far from ideal to establish different selection criteria for each class, based on the FR3D observations, the approach was only fit as a proof of concept.
Instead, we would prefer to get simpler model, ideally with shared limits for most classes or within each basepairing family.
This requires a significant amount of manual effort in selecting appropriate boundaries for each parameter in each class.
Therefore, we need a tool to speed up:

1. Exploration individual instances of each basepair class.
2. Compare the differences between basepairs assigned by FR3D and by a set of our selection criteria;
3. And finally, interactively tuning the selection criteria.

The application which answers these requests to a certain extent is running at https://basepairs.datmos.org (and is attached to the thesis).
This section outlines the basic usage modes, while notes about implementation can be found in @sec:impl-webapp.
It should be noted that it was developed as a tool for internal use with more emphasis on flexibility and comprehensive functionality than a gentle learning curve.
Also note that the application is rather gluttonous when it comes to network bandwidth and system memory (it runs the database in the browser for flexibility).

### Browsing Basepairs

Upon first visit, we are greeted with a home page listing all basepair classes in a table format, reminiscent of <https://doi.org/10.1093/nar/gkf481>.
We are then expected to select a class of interest, either by clicking a table cell, or using the menu at the top of the screen.

When viewing a specific class, we should select the desired "**Data Source**" -- let us select "**FR3D -- Reference Set**" for this demonstration.
This will display images of all basepairs of this class which were reported by FR3D on the Reference Set (@sec:filter).
If we are looking for the most typical cases, we should switch "**Order by**" to the "**Smallest Edge RMSD**".
The edge RMSD is the distance to the most typical basepairs (@sec:opt-params-exemplar), computed as the mean distance of edge (@sec:bp-terminology-lw) atoms, when aligned on all atoms of the second nucleobase.

TODO screenshot - select FR3D
https://basepairs.datmos.org/#tWS-A-A/ds=fr3d-f


If the reference set is insufficiently small, we have the option to select "FR3D -- Entire PDB".
However, since the basepair images need to be pre-generated, the application will now mostly show white squares in their place.
Regardless, we can click on each basepair and a detailed description will appear on the screen.
This modal dialog shows the following information:

* Identity of the basepair (PDB structure ID, chains, residue numbers).
* Two images from different angles, if it was pre-generated.
* A PyMOL script which displays the pair, when pasted to PyMOL command line. We use this to display basepairs which image could not load.
* The calculated parameters (@sec:basepair-params) with a short description.

<!-- TODO: molstar -->

TODO screenshot modalu

### Comparing Sets

When we switch to the **Pairs Selected by New Parameters** data source, we get the option to "**Enable FR3D Comparison**".
We get a union of the basepairs returned by FR3D and the set returned by our proposed criteria.

If the data is still **Ordered by** the **Smallest Edge RMSD**, the screen is unlikely to show anything interesting.
To see the disparities, let us either switch to **Largest Edge RMSD** or select **Show only differences**, instead of **Show all matches**.
Pairs annotated by FR3D and not annotated by our new system are highlighted red, while the ones reported exclusively by us are highlighted in green (it does not mean anything good either).

TODO screenshot diff

At the time of writing and submission of the thesis, the criteria are still subject to change and are dynamically loaded from a shared Google Sheet.
The exact results will be likely different from the ones presented here.

### Editing the Criteria

We will try to improve (or impair) the selection TODO TODO