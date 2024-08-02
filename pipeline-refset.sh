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
pdbcache=~/tmp/pdbcache
# Can be any list of PDB IDs
pdblist="${datadir}/refset/pdbid.list"
# Select the number of threads (in percent of all logical cores or as a fixed number)
# threads=1
threads=100%
# threads=50%

# mkdir -p ./data
# cd ./data
# # TODO pdb download
# cd ..


cd ./scripts
echo "installing Python dependencies"
# poetry install --no-root


# -------------------
echo "finding all close contacts in the refset PDB structures"
mkdir -p $datadir/out/contacts
# rm -rf $datadir/out/contacts # remove old results
# poetry run python pair_finding.py\
#     "--output=$datadir/out/contacts"\
#     "--pdbcache=$pdbcache"\
#     --threads=$threads\
#     $(cat "$pdblist")


# -------------------
echo "analyzing all of the close contacts in refset"
# rm -rf "$datadir/out/contacts-analyzed"
# mkdir -p "$datadir/out/contacts-analyzed"
# poetry run python pairs.py "--override-pair-family=cWW,cWWa,tWW,tWWa,cWH,tWH,cWS,tWS,cHH,cHHa,tHH,cHS,tHS,cSS,tSS,cWB"\
#     --filter --dedupe --postfilter-hb=4.2 --postfilter-shift=2.5\
#     "--pdbcache=$pdbcache"\
#     "--threads=$threads"\
#     "--output=$datadir/out/contacts-analyzed/all.parquet"\
#     "$datadir/out/contacts/"*.parquet\
#     --partition-input-select=0-32/32 # not really needed for refset, but will become handy if you replace $pdblist with a bigger file
cd ..

# -------------------
echo "running FR3D for comparison"
if [ ! -d "./fr3d-python" ]; then
    git clone https://github.com/BGSU-RNA/fr3d-python
fi
cd ./fr3d-python
# TOOD full PDB run
# rm -rf ../data/fr3d-out/
# mkdir -p ../data/fr3d-out/stdout
# mkdir -p ../data/fr3d-out/bp
# find ../data/pdb -type f | poetry --directory ../scripts run parallel -j "$threads" --color --bar --compress --results ../data/fr3d-out/stdout python -m fr3d.classifiers.NA_pairwise_interactions -c basepair_detail -o ../data/fr3d-out/bp '{}'
cd ..

cd ./scripts

# -------------------
echo "analyzing FR3D basepairs"
rm -rf "$datadir/out/fr3d-analyzed"
mkdir -p "$datadir/out/fr3d-analyzed"
poetry run python pairs.py --dedupe\
    "--pdbcache=$pdbcache"\
    "--threads=$threads"\
    "--output=$datadir/out/fr3d-analyzed/fr3d.parquet"\
    "$datadir/fr3d-out/bp/"*_basepair_detail.txt\
    --partition-input-select=0-16/16

cd ..


cd ./webapp
npm install --ci
npm run dev
cd ..
