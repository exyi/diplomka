## Web Gallery of Basepairs {#sec:impl-webapp}

In section [-@sec:tuning-app], we have introduced a web application for browsing the various sets of basepairs (available at https://basepairs.datmos.org/, with the source code attached to this work).
In this section, we would like to uncover some inner working, software design choices and explain advanced usage options.

TODO

### Querying Multiple Classes

```
SELECT * FROM read_parquet('*-filtered')
```

or

```
SELECT * FROM read_parquet('cWW-*')
```


