#/usr/bin/env bash


# Example usage of the provided scripts
# ---
# Calculates the 

# Download

datadir="$(pwd)/data"
pdbcache="${datadir}/pdb"
pdblist="${datadir}/refset/pdbid.list"

# mkdir -p ./data
# cd ./data
# # TODO pdb download
# cd ..


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
    --threads=0\
    $(cat "$pdblist")


# -------------------
echo "analyzing all of the close contacts in refset"
poetry run pairs.py "--override-pair-family=cWW,cWWa,tWW,tWWa,cWH,tWH,cWS,tWS,cHH,cHHa,tHH,cHS,tHS,cSS,tSS,cWB"\
    --filter --dedupe --postfilter-hb=4.2 --postfilter-shift=2.5\
    "--pdbcache=$pdbcache"\
    --threads=0\
    "--output=$datadir/out/contacts-analyzed/all.parquet"\
    "$datadir/out/contacts/"*.parquet\
    --partition-input-select=0-32/32 # not really needed for refset, but will become handy if you replace $pdblist with a bigger file
cd ..

# -------------------
echo "running FR3D for comparison"
if [ ! -d "./fr3d-python" ]; then
    git clone https://github.com/BGSU-RNA/fr3d-python
fi
cd ./fr3d-python
# TOOD full PDB run
mkdir -p ../data/fr3d-out/stdout
mkdir -p ../data/fr3d-out/bp
find -type f ../data/pdb | poetry --directory ../scripts run parallel --color --bar --compress --results ../data/fr3d-out/stdout python -m fr3d.classifiers.NA_pairwise_interactions -o ../data/fr3d-out/bp {}
cd ..

cd ./scripts

# analyze FR3D basepairs
poetry run pairs.py --dedupe\
    "--pdbcache=$pdbcache"\
    --threads=0\
    --output=$datadir/out/$runname.parquet


poetry run pairs.py "--override-pair-family=cWW,cWWa,tWW,tWWa,cWH,tWH,cWS,tWS,cHH,cHHa,tHH,cHS,tHS,cSS,tSS,cWB"\
    --filter --dedupe --postfilter-hb=4.2 --postfilter-shift=2.5\
    "--pdbcache=$pdbcache"\
    --threads=0\
    "--output=$DATADIR/out/$runname.parquet"\
    "$DATADIR/fr3d-exportonly-ID.parquet" "$DATADIR/vsecky-paryIDinv.parquet" "$DATADIR/vsecky-paryID.parquet"\
    $partition_settings 2>&1 | tee $DATADIR/out/$runname-p${BASEPAIRS_PARTITION/\//of}.log

cd ..


cd ./webapp
npm install --ci
npm run dev
cd ..
