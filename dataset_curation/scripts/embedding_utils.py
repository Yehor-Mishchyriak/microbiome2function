from openai import OpenAI
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
import os
from typing import List, Union
import atexit

# MODELS TO CHOOSE:
ESM2 = "facebook/esm2_t6_8M_UR50D" # really light
PROTT5 = "Rostlab/prot_t5_xl_uniref50" # really heavy
SMALL_OPENAI_MODEL = "text-embedding-3-small"
LARGE_OPENAI_MODEL = "text-embedding-3-large"


def get_AAseq_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    return model, tokenizer

# NOTE: this one is not parallelism-safe yet (I didn't bother including threding here as I didn't need to)
# But, if one ever does, be sure to use locks so that a thread writing to the cache acquires a lock and then releases it
# Another thing:
# We load cache once, buffer in a dict, then write exactly one Parquet file at exit;
# The reason we don't use .csv is because it'd serialize embeddings into strings resulting into huge files.
# By the way, that is why fot the final dataset, we'll either use .csv storing paths to .npy's, or parquet again.
# If curios: https://parquet.apache.org/; https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html.
class FreeTXTEmbedder:

    caching_modes = [
    "NOT_CACHING",
    "APPEND",
    "CREATE/OVERRIDE",
    ]

    def __init__(self, api_key, model, cache_file_path=None, caching_mode="NOT_CACHING"):

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache_file_path = cache_file_path
        self.caching_mode = caching_mode

        if cache_file_path:
            if caching_mode == "APPEND":
                
                size = os.path.getsize(cache_file_path)
                if size > 1_000_000_000:
                    raise RuntimeError("Cache >1GB")
                
                df = pd.read_parquet(cache_file_path)
                self.cache_map = dict(zip(df["text"], df["embedding"]))

            elif caching_mode in FreeTXTEmbedder.caching_modes:
                self.cache_map = dict()

            else:
                raise ValueError(f"Invalid 'caching_mode': expected one of {FreeTXTEmbedder.caching_modes}, instead given {caching_mode}")
            
            # save cache at exit
            atexit.register(self.save_cache)

    def get_cache_map(self):
        return self.cache_map

    def lookup_cached(self, s):
        return self.cache_map.get(s)

    def store_in_cache(self, text, emb):
        if self.caching_mode == "NOT_CACHING":
            return

        self.cache_map[text] = emb

    def save_cache(self):
        if not self.cache_file_path or self.caching_mode == "NOT_CACHING":
            return

        df = pd.DataFrame({
            "text": list(self.cache_map.keys()),
            "embedding": list(self.cache_map.values())
        })

        df.to_parquet(self.cache_file_path, index=False, compression="snappy")

    def request_embedding_for(self, inp, batch_size=1000):
        if isinstance(inp, str): inp = [inp]

        n = len(inp)
        results = [None] * n

        to_request, idxs = [], []
        for i, s in enumerate(inp):
            emb = self.lookup_cached(s)
            if emb is not None:
                results[i] = emb
            else:
                to_request.append(s); idxs.append(i) # indices of input strings to request embeddings for

        for start in range(0, len(to_request), batch_size):
            end = start+batch_size

            str_batch = to_request[start:end]
            ind_batch = idxs[start:end]

            resp = self.client.embeddings.create(input=str_batch, model=self.model)
            for s, i, emb in zip(str_batch, ind_batch, resp.data):
                emb = np.array(emb.embedding)

                results[i] = emb
                self.store_in_cache(s, emb)

        return results[0] if n==1 else results


if __name__ == "__main__":
    pass
