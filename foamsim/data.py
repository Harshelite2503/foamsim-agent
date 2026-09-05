"""Experimental reference data: the FoamGPT curated PSP table (bundled snapshot).

    from foamsim.data import load_foamgpt, reference_curve
    df = load_foamgpt()                                   # 951 records, one per composition x test
    ref = reference_curve("epoxy", "glass_microballoon")  # primary compression rows with E or strength vs vf
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "foamgpt_psp.csv"


def load_foamgpt(primary_only: bool = True, unflagged_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(DATA)
    if primary_only:
        df = df[df["data_origin"] == "primary"]
    if unflagged_only:
        df = df[df["flags"].fillna("") == ""]
    return df.reset_index(drop=True)


def reference_curve(matrix_class: str, particle_type: str = "glass_microballoon", test_type: str = "compression",
                    quasi_static: bool = True) -> pd.DataFrame:
    """Rows with particle volume fraction and at least one of modulus/strength/density."""
    df = load_foamgpt()
    d = df[(df.matrix_class == matrix_class) & (df.particle_type == particle_type) & (df.test_type == test_type)]
    if quasi_static:
        d = d[(d.strain_rate_per_s.isna()) | (d.strain_rate_per_s < 1.0)]
    d = d[d.particle_volume_fraction.notna()]
    d = d[d[["modulus_mpa", "strength_mpa", "measured_density_g_cc"]].notna().any(axis=1)]
    cols = ["record_id", "paper_id", "sample_label", "particle_grade", "particle_true_density_g_cc",
            "particle_volume_fraction", "measured_density_g_cc", "modulus_mpa", "strength_mpa", "strain_rate_per_s"]
    return d[cols].sort_values("particle_volume_fraction").reset_index(drop=True)
