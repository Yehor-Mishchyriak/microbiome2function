from openai import OpenAI
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
import os
from typing import List, Union, Tuple
import atexit
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from goatools.obo_parser import GODag

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#                           DENSE EMBEDDINGS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class AAChainEmbedder:
    
    HF_MODELS = {
        "ESM2":   "facebook/esm2_t6_8M_UR50D", # light
        "PROTT5": "Rostlab/prot_t5_xl_uniref50" # heavy
    }

    def __init__(self, model_key: str, device: str = "cpu"):
        if model_key not in self.HF_MODELS:
            raise ValueError(f"model_key must be one of {list(self.HF_MODELS)}")

        self.repo_id   = self.HF_MODELS[model_key]
        self.is_prott5 = (model_key == "PROTT5")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.repo_id, use_fast=False, legacy=False
        )
        self.model = AutoModel.from_pretrained(self.repo_id).to(device).eval()

    @torch.no_grad()
    def embed_sequence(self, seq: str) -> np.ndarray:
        seq = " ".join(seq)
        toks = self.tokenizer(seq, return_tensors="pt").to(self.model.device)
        hidden = self.model(**toks).last_hidden_state # (batch=1, seq_len, hidden_dim), so (seq_len, hidden_dim) basically
        # |  drop special tokens -> (seq_len-2, hidden_dim)
        # v  and pool over residues: (seq_len-2, hidden_dim) -> (hidden_dim)
        emb = hidden[0, 1:-1].mean(dim=0) 
        return emb.cpu().numpy()

    @torch.no_grad()
    def embed_sequences(self, seqs: List[str], batch_size: int = 32) -> List[np.ndarray]:
        if not seqs:
            return []

        seqs = [" ".join(s) for s in seqs]

        out: list[np.ndarray] = []

        for start in range(0, len(seqs), batch_size):
            chunk = seqs[start:start + batch_size]

            toks = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.model.device)

            hidden = self.model(**toks).last_hidden_state
            mask   = toks["attention_mask"]

            for i in range(len(chunk)):
                pos = mask[i].nonzero().squeeze(1)
                body = pos[1:-1] if pos.size(0) > 2 else pos
                emb  = hidden[i, body].mean(dim=0)
                out.append(emb.cpu().numpy())

        return out

class FreeTXTEmbedder:
    MODELS = {
        "SMALL_OPENAI_MODEL": "text-embedding-3-small",
        "LARGE_OPENAI_MODEL": "text-embedding-3-large"
    }
    CACHING_MODES = {"NOT_CACHING", "APPEND", "CREATE/OVERRIDE"}

    def __init__(
        self, api_key: str, model: str,
        cache_file_path: Union[str, None] = None,
        caching_mode: str = "NOT_CACHING",
    ):

        if model not in self.MODELS:
            raise ValueError(f"model must be one of {list(self.MODELS)}")
        if caching_mode not in self.CACHING_MODES:
            raise ValueError(f"caching_mode must be one of {self.CACHING_MODES}")

        self.client = OpenAI(api_key=api_key)
        self.model = FreeTXTEmbedder.MODELS[model]
        self.cache_file_path = cache_file_path
        self.caching_mode = caching_mode

        self.cache_map: dict[str, np.ndarray] = {}
        if cache_file_path and caching_mode == "APPEND" and os.path.exists(cache_file_path):
            df = pd.read_parquet(cache_file_path)
            self.cache_map = {t: np.array(e) for t, e in zip(df["text"], df["embedding"])}

        if cache_file_path and caching_mode != "NOT_CACHING":
            atexit.register(self._save_cache)

    def _save_cache(self):
        if not (self.cache_file_path and self.caching_mode != "NOT_CACHING"):
            return
        df = pd.DataFrame(
            {"text": list(self.cache_map.keys()),
             "embedding": [e.tolist() for e in self.cache_map.values()]}
        )
        df.to_parquet(self.cache_file_path, index=False, compression="snappy")

    def _lookup(self, s: str):
        return self.cache_map.get(s)

    def _store(self, s: str, emb: np.ndarray):
        if self.caching_mode != "NOT_CACHING":
            self.cache_map[s] = emb

    def request_embedding_for(self, inp: Union[str, List[str]], batch_size: int = 1000):
        if isinstance(inp, str):
            inp = [inp]

        out = [None] * len(inp)
        to_send, idx = [], []

        for i, s in enumerate(inp):
            cached = self._lookup(s)
            if cached is not None:
                out[i] = cached
            else:
                to_send.append(s); idx.append(i)

        for start in range(0, len(to_send), batch_size):
            chunk  = to_send[start:start+batch_size]
            ichunk = idx[start:start+batch_size]
            resp   = self.client.embeddings.create(input=chunk, model=self.model)
            for s, i, emb in zip(chunk, ichunk, resp.data):
                vec = np.asarray(emb.embedding, dtype=np.float32)
                out[i] = vec
                self._store(s, vec)

        return out[0] if len(out) == 1 else out

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#                           Multi-hot Encodings
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class MultiHotEncoder:
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def encode(self, sequences: pd.Series):
        # record nans
        nan_mask = sequences.isna()

        # replace NaNs with an empty iterable for now
        # and guarantee everything is an iterable
        tmp = sequences.where(~nan_mask, other="")
        tmp = tmp.map(lambda x: x if isinstance(x, tuple) else (x,) if x else ())

        enc = self.mlb.fit_transform(tmp)

        # convert to object dtype to be able to store NaNs,
        # then restore NaNs where they originally were
        enc = enc.astype(object)
        enc[nan_mask.values] = np.nan

        return {
            "encodings": enc,
            "class_labels": {c: i for i, c in enumerate(self.mlb.classes_)}
        }

class GOEncoder(MultiHotEncoder):
    def __init__(self, obo_path: str):
        super().__init__()
        if not os.path.exists(obo_path):
            raise FileNotFoundError(f"OBO not found: {obo_path}")
        self.godag = GODag(obo_path)

    def _auto_depth(self, series: pd.Series, coverage_target: float = 0.8) -> int:
        depths = [
            self.godag[gid].depth
            for terms in series.dropna()
            for gid in (terms if isinstance(terms, tuple) else (terms,))
            if gid in self.godag
        ]
        if not depths:
            raise ValueError("No valid GO IDs found to compute automatic depth.")
        return int(np.percentile(depths, coverage_target * 100))

    def _collapse_to_depth(self, go_ids: Union[str, tuple], k: int) -> Union[Tuple[str, ...], float]:
        if pd.isna(go_ids):
            return np.nan
        
        if not isinstance(go_ids, tuple):
            go_ids = [go_ids]

        kept = set()
        for gid in go_ids:
            if gid not in self.godag:
                continue
            node = self.godag[gid]
            ancestors = {gid}.union(node.get_all_parents())
            at_k = {n for n in ancestors if self.godag[n].depth == k}
            kept.update(at_k if at_k else {min(ancestors, key=lambda x: self.godag[x].depth)})
        return tuple(kept)

    def process_go(
        self, df: pd.DataFrame, col: str, depth: Union[None, int] = None,
        coverage_target: Union[float, None] = None, inplace: bool = False):

        df = df if inplace else df.copy(deep=True)

        if depth is None:
            if coverage_target is None:
                raise ValueError(
                    "Either `depth` or `coverage_target` must be provided."
                )
            depth = self._auto_depth(df[col], coverage_target)

        collapsed = df.loc[:, col].map(lambda terms: self._collapse_to_depth(terms, depth))

        enc_info = self.encode(collapsed)
        df.loc[:, col] = pd.Series(list(enc_info["encodings"]), index=df.index, dtype=object)
        return df, enc_info["class_labels"]

class ECEncoder(MultiHotEncoder):
    def __init__(self):
        super().__init__()

    def __extract_ec_codes(self, EC: str):
        present_entries = [i for i in EC.split(".") if i.isdigit()]
        return present_entries
    
    def __depth(self, EC: str):
        return len(self.__extract_ec_codes(EC))

    def _auto_depth(self, series: pd.Series, coverage_target: float = 0.8) -> int:
        depths = [
            self.__depth(ec)
            for terms in series.dropna()
            for ec in (terms if isinstance(terms, tuple) else (terms,))
        ]
        if not depths:
            raise ValueError("No valid EC numbers found to compute automatic depth.")
        return int(np.percentile(depths, coverage_target * 100))

    def _collapse_to_depth_helper(self, EC: str, depth: int):
        keep = self.__extract_ec_codes(EC)[:depth]
        return ".".join(keep)
        
    def _collapse_to_depth(self, ECs: Union[str, tuple], depth: int):
        if pd.isna(ECs):
            return np.nan
        if not isinstance(ECs, tuple):
            ECs = [ECs]
        return tuple([self._collapse_to_depth_helper(ec, depth) for ec in ECs])

    def process_ec(self, df: pd.DataFrame, depth: Union[int, None] = None,
                coverage_target: Union[float, None] = None, inplace=False):
        df = df if inplace else df.copy(deep=True)

        col = "EC number"
        if depth is None:
            if coverage_target is None:
                raise ValueError(
                    "Either `depth` or `coverage_target` must be provided."
                )
            depth = self._auto_depth(df[col], coverage_target)

        collapsed = df.loc[:, col].map(lambda terms: self._collapse_to_depth(terms, depth))

        enc_info = self.encode(collapsed)
        df.loc[:, col] = pd.Series(list(enc_info["encodings"]), index=df.index, dtype=object)
        return df, enc_info["class_labels"]


__all__ = [
    "MultiHotEncoder",
    "GOEncoder",
    "FreeTXTEmbedder",
    "AAChainEmbedder",
    "ECEncoder"
]

if __name__ == "__main__":
    pass
