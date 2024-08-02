#/usr/bin/env bash

set -e

# Example usage of the provided scripts
# ---
# 1. Finds all close contacts in the refset PDB structures (and downloads the structures from RCSB as needed)
# 2. Annotates all pairs with the parameters from Section 3.2
# 3. Runs FR3D on the same PDB structures
# 4. Annotates all FR3D base pairs with the same parameters

datadir="$(pwd)/data"
# pdbcache="${datadir}/pdb"
pdbcache="$datadir/pdb"
# Can be any newline-separated list of PDB IDs
pdblist="$datadir/refset/pdbid.list"
# Select the number of threads (in percent of all logical cores or as a fixed number)
# threads=1
# threads=50%
threads=100%


cd ./scripts
echo "installing Python dependencies"
poetry install --no-root


# -------------------
echo "finding all close contacts in the refset PDB structures"
mkdir -p $datadir/out/contacts
rm -rf $datadir/out/contacts # remove old results
poetry run python pair_finding.py\
    "--output=$datadir/out/contacts"\
    "--pdbcache=$pdbcache"\
    --threads=$threads\
    $(cat "$pdblist")


# -------------------
echo "analyzing all of the close contacts in refset"
rm -rf "$datadir/out/contacts-analyzed"
mkdir -p "$datadir/out/contacts-analyzed"
poetry run python pairs.py "--override-pair-family=cWW,cWWa,tWW,tWWa,cWH,tWH,cWS,tWS,cHH,cHHa,tHH,cHS,tHS,cSS,tSS,cWB"\
    --filter --dedupe --postfilter-hb=4.2 --postfilter-shift=2.5\
    "--pdbcache=$pdbcache"\
    "--threads=$threads"\
    "--output=$datadir/out/contacts-analyzed/all.parquet"\
    "$datadir/out/contacts/"*.parquet\
    --partition-input-select=0-32/32 # not really needed for refset, but will become handy if you replace $pdblist with a bigger file

# -------------------
echo "applying the selection criteria to the close contacts - performing the basepair assignment"
poetry run python apply_filter.py "$datadir"/out/contacts-analyzed/all*.parquet --boundaries "$datadir/parameter-boundaries.csv" -o "$datadir/out/assigns.parquet" --best-fit=graph-edges

cd ..

# -------------------
echo "running FR3D for comparison"
if [ ! -d "./fr3d-python" ]; then
    git clone https://github.com/BGSU-RNA/fr3d-python
fi
cd ./fr3d-python
rm -rf "$datadir/fr3d-out/"
mkdir -p "$datadir/fr3d-out/stdout"
mkdir -p "$datadir/fr3d-out/bp"
find "$datadir/pdb" -type f | poetry --directory ../scripts run parallel -j "$threads" --color --bar --compress --results ../data/fr3d-out/stdout python -m fr3d.classifiers.NA_pairwise_interactions -c basepair_detail -o ../data/fr3d-out/bp '{}'
cd ..

cd ./scripts

# -------------------
echo "analyzing FR3D basepairs"
rm -rf "$datadir/out/fr3d-analyzed"
mkdir -p "$datadir/out/fr3d-analyzed"
poetry run python pairs.py --dedupe\
    "--pdbcache=$pdbcache"\
    "--threads=$threads"\
    "--output=$datadir/out/fr3d-analyzed.parquet"\
    "$datadir/fr3d-out/bp/"*_basepair_detail.txt


# -------------------
echo "comparing the assignments (-> data/out/comparison-agg.csv)"
rm -rf "$datadir/out/comparison"
poetry run python compare_sets.py --baseline "$datadir/out/fr3d-analyzed.parquet" --target "$datadir/out/assigns.parquet" --boundaries "$datadir/parameter-boundaries.csv" -o "$datadir/out/comparison-agg.csv" --output-full-diff "$datadir/out/comparison-full-list.parquet"

echo "generating partitioned Parquet files for the web application"

poetry run python pair_distributions.py --residue-directory "$datadir/refset" --skip-kde --skip-plots --reexport=partitioned --output-dir "$datadir/out/reexport-all" "$datadir/out/contacts-analyzed/all*.parquet"
poetry run python pair_distributions.py --residue-directory "$datadir/refset" --skip-kde --skip-plots --reexport=partitioned --output-dir "$datadir/out/reexport-fr3d" "$datadir/out/fr3d-analyzed.parquet"

# copy the partitioned files to the webapp
cd ../webapp

mkdir -p static/tables
for x in "$datadir"/out/reexport-all/*.parquet; do
    cp "$x" static/tables/"$(basename $x .parquet)"-allcontacts.parquet;
done
for x in "$datadir"/out/reexport-fr3d/*.parquet; do
    cp "$x" static/tables/"$(basename $x .parquet)".parquet;
done
npm install --ci
npm run dev
cd ..
