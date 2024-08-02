## Basepair Assignment Procedure {#sec:sw-assignment}

The procedure for assigning basepairs based on the parameters described in @sec:basepair-params requires defining acceptable ranges for each basepairing class.
As a starting point, we can use the ranges from @sec:testing-basepair-params.
With the table of allowed ranges, the assignment procedure is fairly straightforward.

First, we have to identify all potential basepairs to calculate the parameters.
This can, in principle, be all pairs of residues in a given structure, but is advantageous to first prune the list by searching for residues with atoms at ≤ 4 Å distance.
This can be done efficiently using a grid or a [**_k_-d** tree](https://en.wikipedia.org/wiki/K-d_tree) while reduces the algorithmic cost from $\mathcal{O}(n^2)$ to \mathcal{O}(n) or $\mathcal{O}(n\log{}n)$, assuming there can only be a constant number of residues in the vicinity of another one.
We performed this search using NeighborSearch functionality of the [Gemmi library](https://doi.org/10.21105/joss.04200).

For each pair candidate, we calculate the specified parameters for each of the twelve Leontis-Westhof families using the `pairs.py` script.
We omit the `family` column in the input table of all close residues, and substitute it by using the `--override-pair-family=all` option to analyze how each pair fits into all families.
In a third step, we check that all the parameters lie in the defined ranges using the `apply_filter.py` script.


### Whole structure invariants

While the above-mentioned procedure works, we can do further post-processing to make sure that the reported basepairs satisfy the following invariants:

A.  Two residues pair with at most one edge — a single pair does not lie in multiple families.
B.  Each residue has at most one partner at each edge.

Some care is needed regarding disordered atoms (alternative positions).
Rule **A** assumes that the alternative residues are distinct — the alternative position might sensibly lead to a different basepair assignment.
However, rule **B** allows multiple alternatives of a single residue on each edge, effectively treating the alternatives as a single entity.

To enforce the rule in the least disruptive way, we want to keep the _“best”_ subset of pairs and to remove as little as possible.
To quantify that, we need a rudimentary scoring function for basepairs --- which we calculate as the sum of both edge-to-plane distances (@sec:basepair-params-planetoplane) and the H-bond lengths (@sec:basepair-params-hbonds).
Fortunately, this step is not crucial for decent results, and we can afford to simplify it at the cost of being fine-grained.
To avoid giving an unfair advantage to basepairs with a lower number of defined H-bonds, the undefined lengths are filled with value **4.0**, which is usually the H-bond length upper limit.

Scoring of the conflicting basepairs gives us the option to either find a globally optimal selection, or select the best basepairs greedily.
Practical data should contain very little conflicts, so both approaches should yield similar results.
The greedy approach is obviously faster and much easier to implement, but the optimal selection is not hard either, with the right libraries.

In any case, we enforce the rule **A** first without optimizing for rule **B** (i.e., greedily) — it does not seem appropriate to select the optimal basepair family based on its surroundings.
Rule **B** can be reformulated as a problem of maximum weight matching on general graph, which is [solvable in polynomial time ($\mathcal{O}(N^3)$)](https://doi.org/10.1007/s12532-009-0002-8).
In Python, we use the [`algorithms.max_weight_matching` function](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.matching.max_weight_matching.html) from the [NetworkX library](https://networkx.org).
The practical effect of the optimal selection is indeed small; however, it does not slow down the assignment substantially, as the graphs are rather sparse.

<!-- TODO: OMG, některé S páry se nevylučují. I guess ta exkluzivní hrana musí být vodíková vazba (atom), ne celý edge... to asi fixnu, ale už se mi to nechce popisovat -->

<!--
```sql
select pdbid, model, columns('(chain|nr|alt|ins)\d'), count(*), array_agg(family) from './assigned.parquet' group by all having count(*) >1

select pdbid, model, columns('(chain|nr|alt|ins)1'), family[2] as e, count(*) as c, array_agg(family), array_agg(columns('(chain|nr|alt|ins)2')) from './assigned.parquet' group by all having count(*) > 1


select pdbid, model, columns('(chain|nr|alt|ins)\d'), count(*), array_agg(family)
from './assigned.parquet'
WHERE row(pdbid, model, chain1, nr1, alt1, ins1) in (
    select row(pdbid, model, chain1, nr1, alt1, ins1)
    from './assigned.parquet'
    group by pdbid, model, chain1, nr1, alt1, ins1, family[2]
    having count(*) > 1)
group by all
having count(*) >1

```

-->
