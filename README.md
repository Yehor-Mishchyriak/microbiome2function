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
from M2F import (extract_all_accessions_from_dir,
                fetch_save_uniprotkb_batches,
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

accession_nums = extract_all_accessions_from_dir(gene_fam_files_dir, 
                                             pattern=re.compile(r".*_genefamilies\.tsv$"))


out = os.path.join(output_dir, job_name + "_output_dir")
os.makedirs(out, exist_ok=True)

fetch_save_uniprotkb_batches(
    uniref_ids=accession_nums,
    fields=["accession", "ft_domain", "cc_domain",
            "protein_families", "go_f", "go_p",
            "cc_function", "cc_catalytic_activity",
            "ec", "cc_pathway", "rhea", "cc_cofactor", "sequence"],
    batch_size=40_000,
    single_api_request_size=100,
    rps=10,
    save_to_dir=out,
)

print(f"Mined data is available at {out}")
```
Data Cleaning and Feature engineering:
```python
import os
import pandas as pd
import M2F

# ENV & PATHS
raw_data   = os.getenv("RAW_DATA")
output_dir = os.getenv("SAVE_PROCESSED_TO_DIR")
logs_dir   = os.getenv("LOGS_DIR")
job_name   = os.getenv("JOB_NAME")
db_path    = os.getenv("DB")
api_key    = os.getenv("OPENAI_API_KEY")

assert raw_data,   "RAW_DATA env var was not set!"
assert output_dir, "SAVE_PROCESSED_TO_DIR env var was not set!"
assert logs_dir,   "LOGS_DIR env var was not set!"
assert job_name,   "JOB_NAME env var was not set!"
assert api_key,    "OPENAI_API_KEY env var was not set!"
assert db_path,    "DB env var (SQLite cache path) was not set!"

out = os.path.join(output_dir, job_name + "_output_dir")
os.makedirs(out, exist_ok=True)

# LOGGING
M2F.configure_logging(logs_dir)

# MODEL HANDLES
txt_embedder = M2F.FreeTXTEmbedder(
    api_key,
    model="LARGE_OPENAI_MODEL",
    cache_file_path=db_path,
    caching_mode="APPEND",
)

try:
    aa_embedder = M2F.AAChainEmbedder(model_key="esm2_t30_150M_UR50D", device="cuda:0")
except Exception:
    aa_embedder = M2F.AAChainEmbedder(model_key="esm2_t30_150M_UR50D", device="cpu")

# PIPELINE CONFIG
col_names=["Domain [FT]",
        "Domain [CC]",
        "Gene Ontology (molecular function)",
        "Gene Ontology (biological process)",
        "Function [CC]",
        "Catalytic activity",
        "EC number",
        "Pathway",
        "Cofactor",
        "Sequence"
]

apply_norms={"Domain [FT]" : False,
        "Domain [CC]" : True,
        "Gene Ontology (molecular function)" : False,
        "Gene Ontology (biological process)" : False,
        "Function [CC]" : True,
        "Catalytic activity" : False,
        "EC number" : False,
        "Pathway" : True,
        "Cofactor" : False,
        "Sequence" : False
}

txt_embedder = M2F.FreeTXTEmbedder(os.getenv("OPENAI_API_KEY"), model="LARGE_OPENAI_MODEL",
                                   cache_file_path=os.getenv("DB"), caching_mode="APPEND")
aa_embedder = M2F.AAChainEmbedder(model_key="esm2_t30_150M_UR50D", device="cuda:0")


def process_df_inplace(df: pd.DataFrame, *, col_names: list, apply_norms: dict) -> dict:
	# clean
	M2F.clean_cols(df, col_names=col_names, apply_norms=apply_norms, inplace=True)
	# encode
	M2F.embed_ft_domains(df, aa_embedder, inplace=True)
	M2F.embed_AAsequences(df, aa_embedder, inplace=True)
	M2F.embed_freetxt_cols(df, ["Domain [CC]", "Function [CC]", "Catalytic activity", "Pathway"], txt_embedder, inplace=True)
	_, gomf_meta = M2F.encode_go(df, "Gene Ontology (molecular function)", coverage_target=0.8, inplace=True)
	_, gobp_meta = M2F.encode_go(df, "Gene Ontology (biological process)", coverage_target=0.8, inplace=True)
	_, ec_meta = M2F.encode_ec(df, "EC number", inplace=True)
	_, cofactor_meta = M2F.encode_multihot(df, "Cofactor", inplace=True)

	return {"gomf_meta": gomf_meta, "gobp_meta": gobp_meta, "ec_meta": ec_meta, "cofactor_meta": cofactor_meta}


for file in M2F.util.files_from(raw_data):
	# load
	df = pd.read_csv(file)
	# process
	meta = process_df_inplace(df, col_names=col_names, apply_norms=apply_norms)
	# save
	out_pth = out + os.path.basename(file).replace(".csv", ".zip")
	M2F.save_df(df, out_pth, metadata=meta)

print(f"Processed data is available at {out}")
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
