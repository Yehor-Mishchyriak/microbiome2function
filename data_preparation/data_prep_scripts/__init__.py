# pkg:
from .uniprot_parsing import *
# env:
from os import getenv
from dotenv import load_dotenv
load_dotenv()

UNICLUST_TO_UNIREF_MAP = getenv("UNICLUST2UNIREF_MAP")
FETCHED_DATA = getenv("FETCHED_DATA")
RAW_DATA = getenv("RAW_DATA")

__all__ = [
    "UNICLUST_TO_UNIREF_MAP",
    "FETCHED_DATA",
    "RAW_DATA",
    # uniprot_parsing
    "unirefs_from_tsv",
    "retrieve_fields_for_unirefs",
    "process_entries",
    "tsv2df",
    "process_all_tsvs"
]
