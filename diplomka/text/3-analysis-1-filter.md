## Filtration of Structural Data {#sec:filter}

In order to get useful statistics, we need a sufficient quantity of data samples.
However, it is equally important to ensure a high level of data quality to eliminate potential bias.
An example of a problematic bias is shown in @fig:hbonds-hist-filter-comparison-GConly where the H-bond lengths are influenced by restraints.
A degree of unbiased noise may be tolerated, as statistical methods can account for variance.

<!-- Even when working with small sample sizes, the uncertainty arising from the noise can be characterized using confidence intervals.
However, the resulting intervals may be deemed _overconfident_ if the observations are not independent of one another. -->
![Canonical GC pairs, filtered (left) and unfiltered (right). Without the filter, the plot illustrates a large effect from 2.89 Å hydrogen-bond restraints used in model refining.](../img/hbonds-hist-filter-comparison-GConly.png){#fig:hbonds-hist-filter-comparison-GConly}

Since multiple entries in the PDB often share a common refined structure with only varying ligands, we need to remove all redundant structures to avoid the bias.
Moreover, nucleotides must satisfy other quality criteria to be useful for further analysis.
The filtration methodology was prepared by Jiří Černý's laboratory and will be published in a prepared manuscript detailing the entire procedure.
In this project, we utilized the prepared list of about 180 000 nucleotides that were selected from the PDB archive using the quality filters as described in the following section.
Given that the nucleotide filter is rather selective, any basepair with at least one passing nucleotide will be accepted.
We will call this filtered set the **"Reference Set"**.

### Summary of the filtration method

1. Selection of all nucleic acid crystal structures with resolution of 3.5 Å or better
2. Clustering of sequentially redundant structures.
    * Sequences with more than 90% similarity are considered redundant.
    * However, sequences are considered non-redundant, if one is part of a protein complex and the other is not.
3. Selection of the highest quality chain from each cluster of redundancy
    * The score is based on the filter used in [“BGSU RNA score”](https://doi.org/10.1093/nar/gkw453), a weighted sum of resolution, Rfree, clashscore, per-residue value of RSCC, and a fraction of unobserved residues.
4. Selection of residues with sufficiently high quality, defined as:
    * All atoms must have RSCC (real-space correlation coefficient) ≥ 0.7, and backbone atoms must have harmonic of RSCC ≥ 0.8
    * [MolProbity criteria](https://doi.org/10.1002/pro.3330) are satisfied: no steric clashes ≥ 0.5 Å, and no sugar-pucker outliers are allowed.
    * The nucleotide conforms to one of the NtCs ([sugar-phosphate backbone conformations](https://doi.org/10.1093/nar/gkaa383)) with confal score (TODO cite?) ≥ 60% and RMSD ≤ 0.5 Å.

The list of selected residues <!--and detailed description (from Jiří Černý's the manuscript)--> is included as an attachment.

<!-- 

The preparation of a curated reference subset of PDB data involved three steps:
    1. Defining a sequentially non-redundant subset of crystal structures containing nucleic acids;
    2. Finding the highest quality chain in each cluster of homologous sequences;
    3. Applying per-residue quality score cutoffs to the highest quality chains.

To create a sequentially non-redundant subset of structures (Step 1), a list of X-ray PDB entries containing nucleic acids with available reflection data was collected using an NAKB query, returning 8,783 PDB IDs (as of 16 Oct 2022). The sequence information for each chain was obtained from the RCSB PDB using a graphQL query for each ID. All nucleic acid sequences were aligned using the pairwise2.align.localds function of BioPython, employing an extended nucleic acid  substitution matrix. The NAKB query, graphQL query, and alignment code are available in the Supplementary Materials. Only purely DNA or RNA chains were analyzed. The aligned sequences were clustered separately for DNA and RNA based on their sequence dissimilarity. Sequences were considered redundant if they have fewer than three mutations (including gaps/termini) for sequences up to 24 residues, or less than 10% mutations for longer chains. Identical sequences of nucleic acids from NA:protein complexes and from “naked” structures were treated as non-redundant.

The highest quality chain in each cluster of homologous sequences was then identified using a score assigned to each chain (Step 2). The score extended the “BGSU RNA score” (Roll et al., 2016) for a consistent description of DNA and RNA, using validation data downloaded from the RCSB PDB in XML format. The quality score was a weighted combination of resolution (weight 1), Rfree (x 18), clashscore (x 0.05), average per-residue value of 1.000-RSCC (x 8), average per-residue RSR (x 8), and fraction of unobserved residues (x 4). Weights were optimized so that each quality indicator contributed roughly equally to the standard deviation of the composite quality score. The subset of highest quality non-redundant chains belonging to crystal structures with better than 1.8 Å resolution (539 DNA and 206 RNA chains) contained a sufficient number of residues (6,644 DNA and 4,236 RNA) for further analysis; the results supporting this reference set size will be discussed in Section 3.2.

Experience with the development of a similar high-quality reference set for proteins showed that chains of good overall quality almost always contain some extremely poor regions (Williams et al., 2022). Therefore, we decided to implement a residue-level filter to exclude severe errors (Step 3). For this reference set, the most important consideration was to remove cases where a residue is modeled in an incorrect local minimum conformation, resulting in a strain that distorts the covalent geometry. The final reference set was the intersection of the residues that passed the two independent filtering systems described below, one using MolProbity criteria, and one using DNATCO criteria. 

The first filtering system utilized the DNATCO web server (Černý et al., 2020) for the assignment of sugar-phosphate backbone conformation (NtC) to each dinucleotide step (neighboring pair of residues) within a chain. This system is based on the expectation that if all backbone torsion angles, sugar puckers, and the overall shape of a dinucleotide step are close to a known conformational class, the deviations in the covalent geometry of its residues are not too large. Dinucleotide steps assigned to one of the known NtC classes had to fulfil the following criteria: the step confal score should be ≥ 60 (where 100 is the perfect score), backbone atoms harmonic mean real-space correlation coefficient (RSCC) ≥ 0.8, and backbone atoms RMSD ≤ 0.5 Å. Further, if the previous step in the chain was not assigned to a known NtC conformation class, meaning that the more distant 5’-part of the residue in the dinucleotide was less reliable, we used only backbone atoms from C5’ to O3’ in such a residue, otherwise, the residue atoms including the phosphate group were used. This filtering procedure returned 4,336 DNA and 3,082 RNA residues.

The second residue-level filtering system used MolProbity (Williams et al., 2018) and comprised two main components: model-to-map fit and model validation metrics. For model-to-map fit, chains were assessed with phenix.real_space_correlation detail=atom, using .mtz reflection data files provided by the PDB. For a residue to be included in the reference set, all of its member non-H atoms were required to have real-space correlation coefficient (RSCC) ≥ 0.7 and 2mFo-DFc map value ≥ 1.2σ at the atom site. Additionally, the backbone P atom, which carries about twice as many electrons as N/C/O atoms, was required to have 2mFo-DFc map value ≥ 2.4σ.  The B-factor was not used as a filtering criterion, as its treatment was found to be too inconsistent across resolutions and refinement programs. Moreover, for a residue to be included, it was required to have no steric clashes ≥ 0.5 Å (Word et al., 1999). For RNA, residues with sugar pucker outliers (Jain et al., 2015) were also removed. Notably, because this reference was prepared for assessing covalent bond geometry, bond length and bond angle outliers were not used as explicit criteria for filtering. Additionally, non-standard bases and residues with alternate conformations were removed from the reference set, as finding the correct traces through alternate positions is known to be prone to errors (Richardson et al., 2023).

The combined residue-level filtering resulted in 3,202 DNA residues and 2,544 RNA residues modeled with high confidence; the reference set is available in the Supplementary Materials. -->
