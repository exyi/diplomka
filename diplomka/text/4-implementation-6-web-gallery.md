## The [basepairs.datmos.org](https://basepairs.datmos.org){.link-no-footnote} Applicaton {#sec:impl-webapp}

In [section @sec:tuning-app], we have introduced a web application for browsing the various sets of basepairs (available at https://basepairs.datmos.org/, with its source code attached to this work).
In this section, we would like to uncover some inner workings, software design choices and explain some advanced usage options.

### Software design {#sec:impl-webapp-design}

<!-- The trick why re-running FR3D takes 3 hours on 16 CPU cores while we can adjust the parameters interactively is the precomputation and DuckDB -->
The trick why we can adjust the parameters interactively on a website, while re-running FR3D takes about two CPU-days is the pre-computation done in @sec:sw-collection.
And [DuckDB-wasm](https://github.com/duckdb/duckdb-wasm).
We precompute the parameters for all close contacts and perform the filtering later.
The web itself is a purely client-side JavaScript application developed using the [Svelte](https://svelte.dev/) framework.
Rather unconventionally, we run the [DuckDB-wasm database](https://github.com/duckdb/duckdb-wasm) in the web browser to query the statically hosted Parquet tables.
This approach results in a resource-intensive application, but it significantly simplifies the process of implementing flexible filters, even allowing power-users compose their own SQL queries without the necessity for safeguards against [SQL injection](https://en.wikipedia.org/wiki/SQL_injection).

Internal state such as the selected pair, selected data source and filter is persisted into the URL, making it simple to share

#### Basepair images {#sec:impl-webapp-images}

The displayed basepair image are pre-render for the entire reference set using the procedure described in @sec:impl-basepair-img.
The rendered structures are essential for effective browsing of the basepair examples, as the users can quickly identify what does.
Although it is possible to render molecules directly in the browser, loading a bitmap image is an order of magnitude faster even if the molecule is small.
A large fraction of basepairs occur in huge ribosomal structures, making it essentially impossible to render tens of them on one screen even on premium hardware and despite the fact that [MolStar](https://doi.org/10.1093/nar/gkab314) is a comparatively performant renderer.

<!-- TODO: jesti fixnu ten MolStar, tak se samozřejmě pochlubit
However, it makes sense to allow the user to load the structure using [MolStar](https://doi.org/10.1093/nar/gkab314) in the detail modal dialog, especially since we have already paid the steep price for integrating it.

It would most likely significantly help to reduce the loading speed if we used the MolStar Model Server.
This component runs on the server and allows the client to only request certain parts of the molecule.
to je asi blbost popisovat -->

#### Long-term sustainability {#sec:impl-webapp-sustainability}

[Bioinformatics is plagued with unreliable web services](https://doi.org/10.1093/nar/gkaa1125) or services which are no longer operational, [which is partially rooted in the lack of long-term support grants](https://doi.org/10.1371/journal.pcbi.1011920).
We do not have a silver bullet solution to the issue, but we can say we tried to lessen the maintenance cost in the hope of extending the website's longevity.
After the initial batch processing, our service may be hosted any static file web server, and it has been verified to function on both Apache and nginx.
The website can hardly be hosted for free, as it relies on hundreds of gigabytes of pre-rendered basepair images.
However, the absence of any server-side code alleviates the much higher cost of fixing security bugs and updating vulnerable dependencies.
While the cost of fast storage is steadily decreasing, the cost of maintaining a codebase usually increases rapidly with its age.
The web is also an excellent platform for longevity, as it has strong commitment to backward compatibility and allows the same software to work on most hardware that possesses sufficient computational power.

### Querying using SQL

When we switch the filter editor to the **SQL** mode, we gain the freedom to query basepairs or basepair candidates using any expression based on calculated basepair parameters.
The application supports SQL syntax and functions supported by DuckDB 0.9, which includes most of standard SQL with many extensions.

For effective use, we recommend setting a _closest desired filter_ in the **Parameter ranges** mode and then switching to the **SQL** mode.
It will get prepopulated with the query, alleviating the need to remember (or lookup) the internal names of all columns and data sources.
The available columns are displayed under the editor, or in the modal dialog when we click a basepair.

The available tables (i.e., "data sources") are the following are also briefly described under the query editor.
In short, we use the `selectedpair` table to query basepairs of the selected class identified by FR3D, `selectedpair_allcontacts` to get all close contacts (≤ 4.2 Å) and `selectedpair_allcontacts_boundaries` to query basepairs identified by our new set of constraints.
Usually, we append the `_f` suffix to the table name to constrain the query to the reference set (@sec:filter).
Additionally, we can query any other class of basepairs by substituting `selectedpair` for the class name, such as `"tWW-A-A_f"` to select FR3D **tWW A-A** pairs from the reference set (the quotes are necessary).

It is also possible to execute queries against an entire family or any other subset of all pairs by directly calling the `read_parquet(...)` function in the `FROM` clause.
DuckDB allows the argument to be an array or a wildcard, making it easy to query multiple files at once.
We provide the following commonly used examples:

* Single family, FR3D pairs, reference set: `read_parquet('cWW-*-filtered')`
* Single family, FR3D pairs, entire PDB: `read_parquet('cWW-*-unfiltered')`
* Multiple classes, FR3D pairs, entire PDB: `read_parquet(['cWW-G-C', 'cWW-A-U'])`
* Single family, all close contacts, entire PDB: `read_parquet('cWW-á-unfiltered-allcontacts')`
* All data, FR3D pairs, reference set: `read_parquet('*-filtered')`

Not only can we use SQL queries to filter basepairs, we can also perform aggregations.
It is beyond the scope of this work to describe the SQL language versatility, we only note that the application will switch to a tabular view instead of the basepair gallery if the query results do not include the columns necessary to identify basepairs.
This means we can execute queries such as `select count(*) from "cWW-G-C"` and the application is capable of displaying the results.
