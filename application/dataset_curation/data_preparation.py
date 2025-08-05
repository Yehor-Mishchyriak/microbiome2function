import pandas as pd
from scripts import (
    clean_all_cols,
    embed_ft_domains,
    embed_freetxt_cols,
    encode_go,
    encode_multihot,
    save_df
)

for 

df = df.loc[df.notna().all(axis=1)].copy()

df.drop(columns=["Protein families"], inplace=True)
df.drop(columns=["Rhea ID"], inplace=True)

clean_all_cols(df, inplace=True)

embed_ft_domains(df_snippet, aa_embedder, inplace=True)
embed_freetxt_cols(df_snippet, ["Domain [CC]", "Function [CC]", "Catalytic activity", "Pathway"], txt_embedder, inplace=True)
encode_go(df_snippet, "Gene Ontology (molecular function)", coverage_target=0.9, inplace=True)
encode_go(df_snippet, "Gene Ontology (biological process)", coverage_target=0.9, inplace=True)
encode_ec(df_snippet, coverage_target=0.9, inplace=True)
encode_multihot(df_snippet, "Rhea ID", inplace=True)
encode_multihot(df_snippet, "Cofactor", inplace=True)

save_df(df, "/Users/yehormishchyriak/Desktop/BonhamLab/summer2025/microbiome2function/", "testing")