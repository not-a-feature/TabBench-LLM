"""Network-free unit tests for the TCGA loader's parsing helpers."""

from __future__ import annotations

from tabbench_llm.data.loaders.tcga import parse_star_counts, sample_type_to_binary

# A minimal STAR-Counts file: leading `# gene-model` comment, header row, the four
# `N_*` alignment-summary rows, then three gene rows.
_STAR_COUNTS_SAMPLE = (
    "# gene-model: GENCODE v36\n"
    "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second"
    "\ttpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded\n"
    "N_unmapped\t\t\t1000\t1000\t1000\t\t\t\n"
    "N_multimapping\t\t\t2000\t2000\t2000\t\t\t\n"
    "N_noFeature\t\t\t3000\t3000\t3000\t\t\t\n"
    "N_ambiguous\t\t\t4000\t4000\t4000\t\t\t\n"
    "ENSG00000000003.15\tTSPAN6\tprotein_coding\t500\t250\t250\t12.5\t8.0\t9.0\n"
    "ENSG00000000005.6\tTNMD\tprotein_coding\t0\t0\t0\t0.0\t0.0\t0.0\n"
    "ENSG00000000419.13\tDPM1\tprotein_coding\t800\t400\t400\t30.25\t20.0\t22.0\n"
)


def test_parse_star_counts_drops_comment_and_summary_rows():
    s = parse_star_counts(
        _STAR_COUNTS_SAMPLE, value_column="tpm_unstranded", index_column="gene_id"
    )
    assert list(s.index) == ["ENSG00000000003.15", "ENSG00000000005.6", "ENSG00000000419.13"]
    assert s.dtype == "float32"
    assert s["ENSG00000000003.15"] == 12.5
    assert s["ENSG00000000419.13"] == 30.25
    assert not any(g.startswith("N_") for g in s.index)


def test_sample_type_to_binary():
    assert sample_type_to_binary("Solid Tissue Normal") == "Normal"
    assert sample_type_to_binary("Blood Derived Normal") == "Normal"
    assert sample_type_to_binary("Primary Tumor") == "Tumor"
    assert sample_type_to_binary("Metastatic") == "Tumor"
    assert sample_type_to_binary("Additional - New Primary") == "Tumor"
