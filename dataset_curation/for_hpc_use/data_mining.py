from scripts import (recommended_fields_example2, 
                     unirefs_from_multiple_files,
                     process_uniref_batches)
import re
from dotenv import load_dotenv
import os

load_dotenv()

gene_fam_files_dir = os.getenv("SAMPLE_FILES")

accession_nums = unirefs_from_multiple_files(gene_fam_files_dir, 
                                             pattern=re.compile(r".*_genefamilies\.tsv$"))

output_dir = os.getenv("SAVE_DATA_TO_DIR")

process_uniref_batches(
    uniref_ids=accession_nums,
    fields=recommended_fields_example2,
    batch_size=40_000,
    single_api_request_size=100,
    rps=10,
    save_to_dir=output_dir,
    filter_out_bad_ids=True
)

print(f"Mined data is available at {output_dir}")
