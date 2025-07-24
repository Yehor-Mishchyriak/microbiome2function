from transformers import AutoModel, AutoTokenizer

# MODELS TO CHOOSE:
ESM2 = "facebook/esm2_t6_8M_UR50D" # really light
PROTT5 = "Rostlab/prot_t5_xl_uniref50" # really heavy

def get_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=False)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    return model, tokenizer
