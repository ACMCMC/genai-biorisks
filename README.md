# Is My Language Model a Biohazard? — Supplementary Materials

This repository contains the supplementary materials for the (under review) paper "Is My Language Model a Biohazard?".

## Reproducing the results

You'll need to download https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/XML/Compound_000000001_000500000.xml.gz, unzip it, and place it under the folder `pubchem_compounds`.

You'll also need to download https://www.t3db.ca/system/downloads/current/toxins.xml.zip, unzip that too, and place it as `toxins.xml` under a new folder, `t3db`.

Then, you can simply `chmod +x` the `run_full_pipeline.sh` script to make it executable and run it.

The results will be available in the root folder. For convenience, we include them as generated in the repo as well, so there is no need to re-run the pipeline if you just want to inspect the results.