# UniProt Data Engineering Pipeline  

A modular and scalable data engineering toolkit for **protein functional annotation** using UniProtKB data. This package automates:  

- 🔎 **Data mining** from UniProtKB (batched API calls on HPC or locally)  
- 🧹 **Data cleaning & normalization** (regex-based extraction of domains, GO IDs, EC numbers, etc.)  
- ⚙️ **Feature engineering** into numeric representations (dense embeddings & multi-hot encodings)  
- 💾 **Dataset assembly** into graph-ready tensors for ML models (e.g. GNNs)  

This pipeline was built for large-scale **bioinformatics + ML workflows** and is fully HPC-compatible.  

---

## Features  

- **UniProtKB API batching**  
  - Retrieve millions of protein records (`process_uniref_batches`)  
  - Features automatic retries, ID filtering, and per-batch logging  

- **Flexible cleaning utilities**  
  - Regex-based column parsers (`clean_col`, `clean_all_cols`)  
  - Handles nested data: FT domains, GO terms, Rhea IDs, cofactors, etc.  

- **Embedding + encoding**  
  - Dense amino acid sequence embeddings: **ProtT5** & **ESM2** via HuggingFace
  - Free-text functional annotations → OpenAI text embeddings with caching on your machine
  - GO terms, EC numbers, Rhea IDs → multi-hot vectors (with automatic feature space size control)  

- **HPC-first design**  
  - Example **SLURM job script** (`job.sh`)  
  - Timed log rotation and batch-level progress monitoring  

---

## Example Usage
Data Mining:
```python
from microbiome2function.scripts import (recommended_fields_example2, 
                                        unirefs_from_multiple_files,
                                        process_uniref_batches,
                                        configure_logging)
import re
import os


gene_fam_files_dir = os.getenv("SAMPLE_FILES")
output_dir = os.getenv("SAVE_DATA_TO_DIR")
job_name = os.getenv("JOB_NAME")
logs_dir = os.getenv("LOGS_DIR")

assert gene_fam_files_dir, "SAMPLE_FILES env var was not set!"
assert output_dir, "SAVE_DATA_TO_DIR env var was not set!"
assert job_name, "JOB_NAME env var was not set!"
assert logs_dir, "LOGS_DIR env var was not set!"

configure_logging(logs_dir)

accession_nums = unirefs_from_multiple_files(gene_fam_files_dir, 
                                             pattern=re.compile(r".*_genefamilies\.tsv$"))


out = os.path.join(output_dir, job_name + "_output_dir")
os.makedirs(out, exist_ok=True)

process_uniref_batches(
    uniref_ids=accession_nums,
    fields=recommended_fields_example2,
    batch_size=40_000,
    single_api_request_size=100,
    rps=10,
    save_to_dir=out,
    filter_out_bad_ids=True
)

print(f"Mined data is available at {out}")
```
Data Cleaning and Feature engineering:
```python
import pandas as pd
import microbiome2function.scripts as scripts

txt_embedder = scripts.FreeTXTEmbedder(os.getenv("OPENAI_API_KEY"), "SMALL_OPENAI_MODEL",
                                       cache_file_path="testing_cache", caching_mode="CREATE/OVERRIDE")
aa_embedder = scripts.AAChainEmbedder("ESM2")

df = pd.read_csv(os.getenv("FETCHED_DATA"), index_col="Entry")
scripts.clean_all_cols(df, inplace=True)

scripts.embed_ft_domains(df, aa_embedder, inplace=True)
scripts.embed_freetxt_cols(df, ["Domain [CC]", "Function [CC]", "Catalytic activity", "Pathway"], txt_embedder, inplace=True)
scripts.encode_go(df, "Gene Ontology (molecular function)", coverage_target=0.9, inplace=True)
scripts.encode_go(df, "Gene Ontology (biological process)", coverage_target=0.9, inplace=True)
scripts.encode_ec(df, coverage_target=0.9, inplace=True)
scripts.encode_multihot(df, "Rhea ID", inplace=True)
scripts.encode_multihot(df, "Cofactor", inplace=True)
```

## Repository Structure
```graphql
dataset_curation/
├── scripts/
│   ├── mining_utils.py           # API batching & data retrieval
│   ├── cleaning_utils.py         # regex-based text cleaning
│   ├── feature_engineering_utils.py # embeddings & multi-hot encodings
│   ├── embedding_utils.py        # ProtT5, ESM2, OpenAI embedder
│   └── logging_utils.py          # logging setup (HPC-friendly)
├── for_hpc_use/
│   └── data_mining.py            # entrypoint for HPC jobs (example; alter in the way that fits your case)
├── job.sh                        # SLURM job script (example; alter in the way that fits your case)
├── notebooks/
    └── data_processing.ipynb     # notebook example of data preparation (end-to-end)
```

## Why This Matters
This pipeline solves the biggest pain point in protein functional annotation projects:
* Automates the entire data engineering lifecycle
* Scales from hundreds to millions of proteins
* Produces fully numeric datasets ready for deep learning (esp. Graph Neural Networks)
