# TASKS:
## Dataset compilation
1. Feature set identification:
    [] 1.1 Make a script for parsing features of uniref90s from UniProt by their IDs
    [] 1.2 Take a genefamilies .tsv, shuffle its entries and pick 100 random unirefs
    [] 1.3 Parse their features
    [] 1.4 Compute feature statistics: discrete/continuous variable, range, mean, standrad deviation
    [] 1.5 Explore the sample metadata, which can be found on the cluster in /cluster/tufts/bonhamlab/sequencing/

## COMMENTS:
 "Easy" feature completion can be something like "decarboxylases"

 Features to extract:

    Function:
        MOST IMPORTANT:
        cc_function -- General description of protein function. Core for functional annotation tasks.
        cc_catalytic_activity -- Specific enzymatic activities. High confidence functional info.
        ec -- Enzyme Commission numbers. Highly structured, hierarchical function class.
        cc_pathway -- Shows biochemical pathways; helpful for clustering related proteins.
        rhea -- Reaction IDs from Rhea DB (linked to EC); same role as ec, more specific.
        cc_activity_regulation --  Regulatory context is helpful if annotating signaling/regulatory proteins.
        cc_cofactor -- Indicates required cofactors—biochemically informative.
        LESS IMPORTANT, BUT STILL VALUABLE:
        ft_act_site \
        ft_binding   | -- Structural or sequence features hinting at function.
        ft_site     /
        ft_dna_bind -- Specific for transcription factors or DNA-related proteins.
        ph_dependence   \
        temp_dependence  | -- Niche but may help classify environmental adaptations or isoform behavior.
        redox_potential /
    
        * ALL IN ALL:
        function_fields = [
            "cc_function", 
            "cc_catalytic_activity", 
            "ec", 
            "cc_pathway", 
            "rhea", 
            "cc_cofactor", 
            "cc_activity_regulation"
        ]

    Iteraction:
        cc_interaction -- Curated or predicted binary protein-protein interactions
        
        * ALL IN ALL:    
        interaction_fields = [
            "cc_interaction"
        ]

    Gene Ontology:
        go_f -- Covers biochemical activities at the molecular level (e.g. "ATPase activity", "zinc ion binding").
        go_p -- Describes higher-level roles the protein plays (e.g. "DNA replication", "glycolysis").

        * ALL IN ALL:    
        go_fields = [
            "go_f",
            "go_p"
        ]
    
    Family & Domains:
        ft_domain -- Directly linked to function (e.g., kinase, SH3)
        cc_domain -- Complementary to ft_domain
        protein_families -- Often groups proteins by functionally related classes

        * ALL IN ALL:
        fam_n_dom = [
            "ft_domain",
            "cc_domain",
            "protein_families"
        ]