from tqdm import tqdm
from typing import List
import requests
import time
import pandas as pd
from util import flatten
import json

"""
 FEATURE NAMES:
 https://www.uniprot.org/help/return_fields
"""

FIELDS = [
    "cc_function",
    "cc_catalytic_activity",
    "ec",
    "cc_pathway",
    "rhea",
    "cc_cofactor",
    "cc_activity_regulation",
    "cc_interaction",
    "go_f",
    "go_p",
    "ft_domain",
    "cc_domain",
    "protein_families"
]

# This will produce 13 API requests in one call if asking for all the fields, which is a lot.
# I might just pull all the information at once, which I was going to do initially, but then
# I will need to parse it a lot and inefficiently (there's like ~1000 entries in one output dictionary)
def data_from_Uniref90ID(ID: str, *fields) -> dict:
    url = "https://rest.uniprot.org/uniprotkb/search?query={Uniref90_ID}"
    
    extracted_data = dict()

    if fields:
        for i, field in tqdm(enumerate(fields), desc=f"Querying fields for {ID}"):
            if i % 10 == 0:
                time.sleep()
            url_with_field = url + "&fields=" + field
            response = requests.get(url_with_field.format(Uniref90_ID=ID)).text

            try:
                data = json.loads(response)["results"]
            except KeyError:
                print(f"Invalid query for the {field} field")
                data = {}
            
            extracted_data[field] = flatten(data)
    else:
        data = requests.get(url.format(Uniref90_ID=ID)).text
        extracted_data = flatten(data)

    return extracted_data

def uniprot_resp2pandas():
    pass
