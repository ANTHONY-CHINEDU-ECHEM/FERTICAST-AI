# Data

`make data` writes `data/raw/ivf_cycles.csv`, a synthetic register of 120,000 treatment cycles (plus duplicated
submissions that the cleaning step removes). Nothing in this folder describes a real person.

To run the pipeline on the public HFEA anonymised register instead, download the register from the HFEA website,
convert it with `ferticast.data.hfea_adapter.load_hfea_register`, save the result to `data/raw/ivf_cycles.csv`
and run `make train`. The register does not contain hormone assays, BMI or semen parameters, so those inputs are
imputed and the model effectively becomes an age and history model on that source.
