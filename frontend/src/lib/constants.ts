export const TE_COLUMNS = [
  "TE_108T", "TE_12T", "TE_A2780", "TE_A549", "TE_BJ", "TE_BRx-142",
  "TE_C643", "TE_CRL-1634", "TE_Calu-3", "TE_Cybrid_Cells", "TE_H1-hESC",
  "TE_H1933", "TE_H9-hESC", "TE_HAP-1", "TE_HCC_tumor",
  "TE_HCC_adjancent_normal", "TE_HCT116", "TE_HEK293", "TE_HEK293T",
  "TE_HMECs", "TE_HSB2", "TE_HSPCs", "TE_HeLa", "TE_HeLa_S3", "TE_HepG2",
  "TE_Huh-7.5", "TE_Huh7", "TE_K562", "TE_Kidney_normal_tissue", "TE_LCL",
  "TE_LuCaP-PDX", "TE_MCF10A", "TE_MCF10A-ER-Src", "TE_MCF7", "TE_MD55A3",
  "TE_MDA-MB-231", "TE_MM1.S", "TE_MOLM-13", "TE_Molt-3", "TE_Mutu",
  "TE_OSCC", "TE_PANC1", "TE_PATU-8902", "TE_PC3", "TE_PC9",
  "TE_Primary_CD4+_T-cells",
  "TE_Primary_human_bronchial_epithelial_cells", "TE_RD-CCL-136",
  "TE_RPE-1", "TE_SH-SY5Y", "TE_SUM159PT", "TE_SW480TetOnAPC", "TE_T47D",
  "TE_THP-1", "TE_U-251", "TE_U-343", "TE_U2392", "TE_U2OS", "TE_Vero_6",
  "TE_WI38", "TE_WM902B", "TE_WTC-11", "TE_ZR75-1",
  "TE_cardiac_fibroblasts", "TE_ccRCC", "TE_early_neurons",
  "TE_fibroblast", "TE_hESC", "TE_human_brain_tumor",
  "TE_iPSC-differentiated_dopamine_neurons", "TE_megakaryocytes",
  "TE_muscle_tissue", "TE_neuronal_precursor_cells", "TE_neurons",
  "TE_normal_brain_tissue", "TE_normal_prostate", "TE_primary_macrophages",
  "TE_skeletal_muscle",
] as const;

export const NUM_TE = TE_COLUMNS.length; // 78

export function displayName(col: string): string {
  return col.replace(/^TE_/, "").replace(/_/g, " ");
}

export const VALID_AA = new Set("ACDEFGHIKLMNPQRSTVWYX");

export function validateProtein(seq: string): string | null {
  const upper = seq.toUpperCase().replace(/\s/g, "");
  for (const ch of upper) {
    if (!VALID_AA.has(ch)) {
      return `Invalid amino acid character: '${ch}'`;
    }
  }
  return null;
}
