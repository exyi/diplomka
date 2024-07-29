#!/usr/bin/env bash
export PYTHONUNBUFFERED=1

systemd-run --scope -p MemoryMax=30000M --user\
    python3 ./pairs.py\
    /mnt/Aldabrachelys/test/vsecky-paryID.parquet /mnt/Aldabrachelys/test/vsecky-paryIDinv.parquet /mnt/Aldabrachelys/test/pdb/analyzed-bonds/fr3d-exportonly-ID.csv.parquet\
    --override-pair-family cWW,tWW,cWH,tWH,cWS,tWS,cHH,tHH,cHS,tHS,cSS,tSS\
    --pdbcache /mnt/Aldabrachelys/test/pdb/s\
    --dedupe --filter --postfilter-hb=4.2 --postfilter-shift=2.5\
    --output /mnt/Aldabrachelys/test/pdb/analyzed-bonds/PDBallcontacts+fr3d-alltypes.parquet\
    --partition-input-select=$1/40\
    --threads 24 2>&1 | tee /mnt/Aldabrachelys/test/pdb/analyzed-bonds/PDBallcontacts+fr3d-alltypes-$1.log
    #--reference-basepairs /home/exyi/code/rna-ml/pairclusters/out/plots-KDE-yawpitchroll/output.json \
