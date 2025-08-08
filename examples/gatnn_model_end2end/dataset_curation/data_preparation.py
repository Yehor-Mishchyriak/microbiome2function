from M2F import (clean_cols,
             FreeTXTEmbedder,
            AAChainEmbedder,
            embed_ft_domains,
            embed_AAsequences,
            embed_freetxt_cols,
            encode_go,
            encode_ec,
            encode_multihot)
import os

VAR = os.getenv("VAR")
output_dir = os.getenv("SAVE_DATA_TO_DIR")
job_name = os.getenv("JOB_NAME")
logs_dir = os.getenv("LOGS_DIR")

assert VAR, "VAR env var was not set!"

M2F.configure_logging(getenv("LOGS_DIR"))

clean_df = clean_cols(
        df,

        col_names=["Domain [FT]",
                    "Domain [CC]",
                    "Gene Ontology (molecular function)",
                    "Gene Ontology (biological process)",
                    "Function [CC]",
                    "Catalytic activity",
                    "EC number",
                    "Pathway",
                    "Cofactor",
                    "Sequence"],

        apply_norms={"Domain [FT]" : False,
                "Domain [CC]" : True,
                "Gene Ontology (molecular function)" : False,
                "Gene Ontology (biological process)" : False,
                "Function [CC]" : True,
                "Catalytic activity" : False,
                "EC number" : False,
                "Pathway" : True,
                "Cofactor" : False,
                "Sequence" : False}
)

txt_embedder = FreeTXTEmbedder(getenv("OPENAI_API_KEY"), model="LARGE_OPENAI_MODEL",
                                   cache_file_path="test.db", caching_mode="APPEND")
aa_embedder = AAChainEmbedder()

embed_ft_domains(clean_df_portion, aa_embedder, inplace=True)
embed_AAsequences(clean_df_portion, aa_embedder, inplace=True)
embed_freetxt_cols(clean_df_portion, ["Domain [CC]", "Function [CC]", "Catalytic activity", "Pathway"], txt_embedder, inplace=True)
encode_go(clean_df_portion, "Gene Ontology (molecular function)", coverage_target=0.9, inplace=True)
encode_go(clean_df_portion, "Gene Ontology (biological process)", coverage_target=0.9, inplace=True)
encode_ec(clean_df_portion, "EC number", coverage_target=0.9, inplace=True)
encode_multihot(clean_df_portion, "Cofactor", inplace=True)

clean_df_portion.sort_values(by="Entry", inplace=True)
clean_df_portion.sort_index(axis=1, inplace=True)