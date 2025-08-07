# builtins:
import re
import os
from typing import Union, List, Tuple, Dict, Any, Optional

# third-party:
import pandas as pd
import numpy as np
import zarr
from zarr.core.dtype.npy.bytes import RawBytes
from .embedding_utils import (FreeTXTEmbedder,
                              AAChainEmbedder,
                              GOEncoder,
                              ECEncoder,
                              encode_multihot)

# local:
from . import util


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# *-----------------------------------------------*
#                      UTILS
# *-----------------------------------------------*

def max_pool(embeddings: List[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("Cannot max pool empty list of embeddings")
    
    if len({emb.shape for emb in embeddings}) != 1:
        raise ValueError(f"Embedding shapes differ: {[emb.shape for emb in embeddings]}")

    if len(embeddings) == 1:
        return embeddings[0]
    
    # calculate L2 norm for each embedding
    norms = [np.linalg.norm(emb) for emb in embeddings]
    
    # return the embedding with maximum norm
    max_idx = np.argmax(norms)
    return embeddings[max_idx]

def vals2embs_map(df: pd.DataFrame, col: str, embedder: Union[AAChainEmbedder, FreeTXTEmbedder], batch_size: int):
    unique_vals = df[col].dropna().unique()
    vals = [item
            for val in unique_vals
            for item in val
            if item
        ] # flatten
    if not vals:
        return {} 
    val2emb_map = dict(zip(vals, embedder.embed_sequences(vals, batch_size)))
    return val2emb_map

# there's a bunch of harmless warnings caused by zipfile lib,
# and one zarr warning regarding a dtype I used for strings -- also harmless
@util.suppress_warnings(zarr.core.dtype.common.UnstableSpecificationWarning, UserWarning)
def save_df(df: pd.DataFrame, name: str, metadata: Optional[dict] = None) -> None:
    # ---------define-helpers---------
    def encode_strings(strings: np.ndarray) -> Tuple[str, np.ndarray]:
        max_len = max(len(s) for s in strings)
        # dtype='S⟨max_len⟩' == “fixed-length bytes” (1 byte per char)
        dtype = f"S{max_len}"
        data = np.asarray(strings, dtype=dtype)
        return dtype, data

    def flatten_offset(tuples: tuple) -> Tuple[np.ndarray, np.ndarray]:
        flat_vals = np.fromiter((x for tup in tuples for x in tup), dtype='int32')
        lengths = np.array([len(t) for t in tuples], dtype='int32')
        # offsets assume an extra entry at end for slicing convenience
        offsets = np.empty(len(tuples)+1, dtype='int32')
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        return np.array(flat_vals, dtype=np.uint16), np.array(offsets, dtype=np.uint16)
    # --------------------------------

    with zarr.storage.ZipStore(f"{name}.zip", mode="w") as store:
        root = zarr.group(store, overwrite=True)
        
        # -----save-accession-pointers----
        df = df.sort_values(by="Entry").copy()
        dtype, data_bytes = encode_strings(df["Entry"].to_numpy())
        root.create_array('accessions', data=data_bytes, compressors=zarr.codecs.BloscCodec(cname="zlib", clevel=3, shuffle=zarr.codecs.BloscShuffle.noshuffle))
        # --------------------------------

        # ---------save-cols-data---------
        for col_name in sorted([col for col in df.columns if col != "Entry"]):
            col_storage = root.create_group(col_name)
            notna_mask = ~df[col_name].isna()
            # filter out missing data
            notna_accessions = df["Entry"][notna_mask]
            notna_vals = df[col_name][notna_mask]
            if notna_vals.empty:
                col_storage.attrs["is_empty"] = True
                continue
            else:
                col_storage.attrs["is_empty"] = False

            accession_dtype, data_bytes = encode_strings(notna_accessions.to_numpy())
            col_storage.create_array('accessions', data=data_bytes,
                                    compressors=zarr.codecs.BloscCodec(cname="zlib", clevel=3, shuffle=zarr.codecs.BloscShuffle.noshuffle))

            if isinstance(notna_vals.iloc[0], tuple):
                flat_vals, offsets = flatten_offset(notna_vals)
                col_storage.create_array('flat_vals', data=flat_vals,
                                        compressors=zarr.codecs.BloscCodec(cname="lz4", clevel=2, shuffle=zarr.codecs.BloscShuffle.bitshuffle))
                col_storage.create_array('offsets', data=offsets,
                                        compressors=zarr.codecs.BloscCodec(cname="lz4", clevel=2, shuffle=zarr.codecs.BloscShuffle.bitshuffle))

            elif isinstance(notna_vals.iloc[0], np.ndarray):
                data = np.vstack(notna_vals, dtype=np.float32)
                col_storage.create_array(name="data", data=data,
                                        compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle))

            else:
                raise ValueError(f"The dataframe contains unsupported dtype: {type(notna_vals.iloc[0])}. Expected: 'tuple' or 'ndarray'")

        # -----------add-metadata---------
        if metadata is None:
            return
        for attr, desc in metadata.items():
            root.attrs[attr] = desc
        # --------------------------------

def load_df(path: str) -> pd.DataFrame:
    if not path.endswith(".zip"):
        path += ".zip"
    store = zarr.storage.ZipStore(path, mode="r")
    root = zarr.open_group(store, mode="r")

    def decode_strings(b: np.ndarray) -> np.ndarray:
        return np.char.decode(b, encoding="ascii")

    full_acc = decode_strings(root["accessions"][:])
    df = pd.DataFrame({"Entry": full_acc})

    for col_name in root.group_keys():
        grp = root[col_name]
        is_empty = grp.attrs.get("is_empty")

        if is_empty:
            df[col_name] = np.nan
            continue

        col_acc = decode_strings(grp["accessions"][:])

        if {"flat_vals", "offsets"} <= set(grp.array_keys()):
            flat = grp["flat_vals"][:].astype(int)
            offs = grp["offsets"][:].astype(int)
            values = [
                tuple(flat[offs[i]:offs[i + 1]])
                for i in range(len(col_acc))
            ]

        elif "data" in grp.array_keys():
            data   = grp["data"][:].astype(np.float32)
            values = [row for row in data]

        else:
            raise ValueError(
                f"Unknown layout in column '{col_name}': {grp.array_keys()}"
            )

        mapping = dict(zip(col_acc, values))
        df[col_name] = df["Entry"].map(mapping)

    df.attrs.update(dict(root.attrs))
    df.sort_index(axis=1, inplace=True)

    return df

def empty_tuples_to_NaNs(df: pd.DataFrame, inplace=False) -> None:
    if not inplace:
        df = df.copy(deep=True)
    df.loc[:, :] = df.map(lambda x: np.nan if x == () else x)
    return df

# *--------------------------------------------------------*
# cc_domain, cc_function, cc_catalytic_activity, cc_pathway
# *--------------------------------------------------------*

def embed_freetxt_cols(df: pd.DataFrame, cols: List[str], embedder: FreeTXTEmbedder,
                    batch_size: int = 1000, inplace=False) -> pd.DataFrame:
    if not inplace:
        df = df.copy(deep=True)
    for col in cols:
        embedding_map = vals2embs_map(df, col, embedder, batch_size)
        df.loc[:, col] = df.loc[:, col].map(lambda entry: max_pool([embedding_map[s] for s in entry])
                                            if entry else np.nan)
    return df

# *-----------------------------------------------*
#               ft_domain, sequence
# *-----------------------------------------------*

def _domain_aa_ranges(domains: Tuple[str]) -> List[Tuple[int,int]]:
    ranges = []
    for d in domains:
        m = re.match(r"^(\d+)\.\.(\d+)$", d)
        if not m:
            continue
        start, end = map(int, m.groups())
        ranges.append((start - 1, end))
    return ranges

def _get_domain_sequences(domains: Tuple[str, ...], full_seq: Tuple[str]) -> List[str]:
    # Note: need to do full_seq[0] because it's a singleton tuple
    return [full_seq[0][s:e] for s, e in _domain_aa_ranges(domains)]

def embed_ft_domains(df: pd.DataFrame, embedder: AAChainEmbedder,
                     batch_size: int = 128, inplace: bool = False) -> pd.DataFrame:
    if not inplace:
        df = df.copy(deep=True)

    df.loc[:, "tmp_domain_seqs"] = df.apply(
        lambda row: tuple(_get_domain_sequences(row["Domain [FT]"], row["Sequence"])), axis=1
    )

    embedding_map = vals2embs_map(df, "tmp_domain_seqs", embedder, batch_size)

    df.loc[:, "Domain [FT]"] = df["tmp_domain_seqs"].map(
        lambda entry: max_pool([embedding_map[s] for s in entry]) if entry else np.nan
    )
    df.drop(columns="tmp_domain_seqs", inplace=True)
    
    return df

def embed_AAsequences(df: pd.DataFrame, embedder: AAChainEmbedder,
                      batch_size: int = 128, inplace: bool = False) -> pd.DataFrame:
    if not inplace:
        df = df.copy(deep=True)

    embedding_map = vals2embs_map(df, "Sequence", embedder, batch_size)
    df.loc[:, "Sequence"] = df["Sequence"].map(lambda s: embedding_map[s[0]] if s else np.nan)

    return df

# *-----------------------------------------------*
#                   go_mf & go_bp
# *-----------------------------------------------*

_go_enc = GOEncoder(os.path.join(SCRIPT_DIR, "dependencies", "go-basic.obo"))
encode_go = _go_enc.encode_go

# *-----------------------------------------------*
#                       ec
# *-----------------------------------------------*

_ec_enc = ECEncoder()
encode_ec = _ec_enc.encode_ec


__all__ = [
    "embed_ft_domains",
    "embed_AAsequences",
    "embed_freetxt_cols",
    "empty_tuples_to_NaNs",
    "encode_go",
    "encode_ec",
    "encode_multihot",
    "save_df",
    "load_df"
]

if __name__ == "__main__":
    pass
