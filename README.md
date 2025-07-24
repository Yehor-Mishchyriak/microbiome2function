# TASKS:
feature_preprocessing:
[] "Domain [FT]" :
    (X) regex Domain AA sequence extraction -> () embedding via ProtT5

[] "Domain [CC]" :
    (X) regex Free-text extraction -> () embedding via openai text-embedding-3-large

[] "Protein families":
    (X) regex fam extraction -> () dimensionaly reduction via Pfam clans / superfamily hierarchy -> () embed as k-hot

[] "Gene Ontology (molecular function)" :
    (X) regex extract GO -> () dimensionaly reduction via GO DAG -> () embed as k-hot"

[] "Gene Ontology (biological process)" :
    (X) regex extract GO -> () dimensionaly reduction via GO DAG -> () embed as k-hot"

[] "Function [CC]" : 
    (X) regex Free-text extraction -> () embedding via openai text-embedding-3-large; preserve vector representation of each function for KNN style prediction down the road"

[] "Catalytic activity" :
    (x) regex Extract reactions -> () embedding via openai text-embedding-3-large"

[] "EC number" : 
    (X) regex extract ec # -> () dimensionaly reduction via EC hierarchy -> () embed as k-hot"

[] "Pathway" :
    (X) regex extract pathway -> () dimensionaly reduction via KEGG/Rhea pathway hierarchy -> () embed as k-hot"

[] "Rhea ID" : 
    (X) regex extract rhea id -> () embed as k-hot"

[] "Cofactor" : 
    (x) regex extract cofactor name -> () embed as k-hot".

TIPS:
Batch & cache your OpenAI calls to avoid rate-limit slowdowns.

When you assemble your final feature vector, you might want to project each embedding down (e.g. via a small linear layer)
so your GAT doesn't explode in dimensionality.

Keep an eye on missing data: do masked features + learnable imputation.