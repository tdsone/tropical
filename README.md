# tropical
tropical is an amalgamate of ML models to optimize RNA sequences for high tissue specificity

## Ideas
- You need cell type specific expression for your medication to be safe.
    - Especially important to avoid off-target expression.
    - Q1: what are common off-target cell types where LNPs go? 
        - LNPs often accumulated in the liver (?) (Source?) 
        - NewLimit has three programs: metabolism with hepatocytes, vascular with endothelial cells and immunology with T cells (any or CD8+/CD4+/T reg/...)


# Background

- What are known cases where sequence drives cell-type specificity?
    - In the liver, a specific micro-RNA called miRNA-122 is highly abundant. By incorporating the "reverse complement" of its sequence into the 3’ UTR of your payload mRNA, miRNA-122 will bind to the payload and trigger its rapid degradation. Because dendritic cells lack miRNA-122, the mRNA remains stable and produces high protein levels there, while remaining virtually silent in the liver where miRNA-122 is abundant. https://en.wikipedia.org/wiki/MiR-122

- From COVID 19 vaccine report: https://www.ema.europa.eu/en/documents/assessment-report/comirnaty-epar-public-assessment-report_en.pdf
    > Radioactivity was detected in most tissues from the first time point (0.25 h) and results support that injections site and the liver are the major sites of distribution.
    
    > Low levels of radioactivity were detected in most tissues, with the greatest levels in plasma observed 1-4 hours post-dose.

    > Over 48 hours, distribution was mainly observed to liver, adrenal glands, spleen and ovaries, with maximum concentrations observed at 8-48 hours post-dose. Total recovery (% of injected dose) of radiolabeled LNP+modRNA outside the injection site was greatest in the liver (up to 21.5%) and was much less in spleen (≤1.1%), adrenal glands (≤0.1%) and ovaries (≤0.1%).

    - So biodistribution changes over time -> If you don't go for liver delivery, the detargeting the liver is probably no.1 prio; I'm not sure if adrenal glands is just on this list because it was foreign epitopes or if there is a natural preference of LNPs for adrenal glands

- When an LNP enters the bloodstream, it doesn't stay a "naked" nanoparticle. Blood proteins immediately stick to its surface, forming what is called a "protein corona." The most prominent protein that binds to LNPs is Apolipoprotein E (ApoE). With ApoE as a "label" it guides the LNP towards the adrenal glands which require large amounts of cholesterol to make hormones.
- The injection site itself will have a lot of LNP exposure too.
    Muscle Cells (Myocytes): Since the injection is intramuscular, myocytes are the most abundant cells in the immediate vicinity. They take up a massive portion of the LNPs.

    Tissue-Resident Immune Cells: Your muscle tissue is constantly patrolled by local immune cells, primarily macrophages and dendritic cells. These cells are literally designed to sample their environment and swallow foreign particles, so they aggressively gorge on the local LNPs.

    Fibroblasts: These are the structural cells that make up the connective tissue holding your muscle fibers together. They also readily absorb the nearby LNPs.

    Endothelial Cells: These are the cells that line the tiny capillaries and blood vessels running through the muscle.