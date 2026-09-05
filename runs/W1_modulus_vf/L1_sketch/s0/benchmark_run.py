"""Compressive modulus of epoxy / 3M K46 glass-microballoon syntactic foam vs particle volume fraction.

Analytical micromechanics with the foamsim toolkit:
  - Mori-Tanaka with hollow-sphere equivalent particles (HP-MT)
  - Differential scheme (HP-DS)
  - Hashin-Shtrikman bounds as a sanity band
plus self-checks and a comparison with the bundled FoamGPT experimental data.

Units: MPa (moduli), g/cm^3 (density), vf = volume fraction of particles INCLUDING their hollow cores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from foamsim import MATERIALS, hollow_particle
from foamsim.micromechanics import (
    RCP,
    density,
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
)
from foamsim.data import reference_curve

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def main() -> None:
    matrix = MATERIALS["epoxy"]
    particle = hollow_particle("K46")

    print("=== Constituents ===")
    print(f"matrix   epoxy : E={matrix.E:.0f} MPa, nu={matrix.nu:.3f}, rho={matrix.rho:.3f} g/cm^3")
    print(f"particle K46   : shell E={particle.shell.E:.0f} MPa, nu={particle.shell.nu:.3f}, "
          f"rho_shell={particle.shell.rho:.3f} g/cm^3")
    print(f"                 eta (inner/outer radius ratio) = {particle.eta:.4f}, "
          f"diameter = {particle.diameter_um} um")
    assert 0.0 <= particle.eta < 1.0, "eta must be in [0,1)"

    # --- sweep -------------------------------------------------------------
    # vf capped at 0.60: RCP for monodisperse spheres is 0.64 and >~0.55 already
    # needs polydisperse packing, so the top of the range is optimistic.
    vfs = np.linspace(0.0, 0.60, 13)
    rows = []
    for vf in vfs:
        mt = hollow_particle_mori_tanaka(matrix, particle, vf)
        ds = hollow_particle_differential(matrix, particle, vf)
        hs = hashin_shtrikman_bounds(matrix, particle, vf)
        rows.append({
            "vf": vf,
            "E_MT_mpa": mt.E,
            "E_DS_mpa": ds.E,
            "E_HS_lo_mpa": hs["E_lo"],
            "E_HS_hi_mpa": hs["E_hi"],
            "nu_MT": mt.nu,
            "rho_g_cc": density(matrix, particle, vf),
            "specific_E_MT": mt.E / density(matrix, particle, vf),
        })
    sweep = pd.DataFrame(rows)

    print("\n=== HP-MT / HP-DS modulus vs volume fraction (E in MPa) ===")
    print(sweep.to_string(index=False, float_format=lambda x: f"{x:9.3f}"))
    sweep.to_csv("modulus_vs_vf.csv", index=False)
    print("\n(wrote modulus_vs_vf.csv)")

    # --- self checks -------------------------------------------------------
    print("\n=== Self-checks ===")
    e0 = sweep.iloc[0]
    assert abs(e0.E_MT_mpa - matrix.E) < 1e-6 * matrix.E, "vf=0 must recover the matrix modulus"
    assert abs(e0.rho_g_cc - matrix.rho) < 1e-9, "vf=0 must recover the matrix density"
    print(f"vf=0 recovers the matrix: E={e0.E_MT_mpa:.1f} MPa, rho={e0.rho_g_cc:.3f} g/cm^3  [OK]")

    inside = ((sweep.E_MT_mpa >= sweep.E_HS_lo_mpa - 1e-6) & (sweep.E_MT_mpa <= sweep.E_HS_hi_mpa + 1e-6)
              & (sweep.E_DS_mpa >= sweep.E_HS_lo_mpa - 1e-6) & (sweep.E_DS_mpa <= sweep.E_HS_hi_mpa + 1e-6))
    assert inside.all(), "every estimate must lie inside the Hashin-Shtrikman bounds"
    print("both estimates lie inside the HS bounds at every vf  [OK]")
    print(f"RCP limit for monodisperse spheres = {RCP:.2f}; sweep stops at vf={vfs[-1]:.2f}  [OK]")

    trend = "increases" if sweep.E_MT_mpa.iloc[-1] > sweep.E_MT_mpa.iloc[0] else "decreases"
    print(f"K46 is a stiff particle -> E {trend} with vf "
          f"({sweep.E_MT_mpa.iloc[0]:.0f} -> {sweep.E_MT_mpa.iloc[-1]:.0f} MPa)")

    # --- comparison with experiment ---------------------------------------
    print("\n=== Comparison with FoamGPT experimental data (epoxy / glass microballoons) ===")
    ref = reference_curve("epoxy", "glass_microballoon")
    ref = ref[ref.modulus_mpa.notna()]
    grade = ref.particle_grade.astype(str)
    is_k46 = grade.str.upper().str.contains("K46", na=False)
    # Rows naming K46 alongside other grades are layered / functionally graded specimens,
    # not a monolithic epoxy + K46 foam, so they are reported separately rather than mixed in.
    other_grades = grade.str.upper().str.contains("S22|S32|S38|S60|K1 |K15|K20|K25|K37|IM16|IM30", na=False)
    k46_pure = ref[is_k46 & ~other_grades]

    def compare(subset: pd.DataFrame, label: str) -> pd.DataFrame:
        if subset.empty:
            print(f"\n-- {label}: no rows --")
            return subset
        c = subset[["record_id", "particle_grade", "particle_volume_fraction", "modulus_mpa"]].copy()
        c["E_MT_mpa"] = [hollow_particle_mori_tanaka(matrix, particle, float(v)).E
                         for v in c.particle_volume_fraction]
        c["E_DS_mpa"] = [hollow_particle_differential(matrix, particle, float(v)).E
                         for v in c.particle_volume_fraction]
        c["rel_err_MT_pct"] = 100.0 * (c.E_MT_mpa - c.modulus_mpa) / c.modulus_mpa
        bands = [hashin_shtrikman_bounds(matrix, particle, float(v)) for v in c.particle_volume_fraction]
        c["in_HS_band"] = [b["E_lo"] <= m <= b["E_hi"] for b, m in zip(bands, c.modulus_mpa)]
        c["subset"] = label
        c["particle_grade"] = c.particle_grade.str.slice(0, 60)
        print(f"\n-- {label}: {len(c)} points --")
        print(c.drop(columns=["subset"]).to_string(index=False, float_format=lambda x: f"{x:9.2f}"))
        print(f"   MAPE  HP-MT = {np.mean(np.abs(c.rel_err_MT_pct)):.1f}% , "
              f"HP-DS = {np.mean(np.abs(100.0 * (c.E_DS_mpa - c.modulus_mpa) / c.modulus_mpa)):.1f}% ; "
              f"inside HS band: {int(c.in_HS_band.sum())}/{len(c)}")
        return c

    print(f"{len(ref)} epoxy/GMB compression rows with a reported modulus; "
          f"{int(is_k46.sum())} mention K46, of which {len(k46_pure)} are K46-only specimens.")
    all_cmp = compare(ref, "all epoxy / glass-microballoon grades")
    k46_cmp = compare(k46_pure, "K46-only specimens")
    mixed = compare(ref[is_k46 & other_grades], "K46 + other grades (layered / FGM - not comparable)")
    pd.concat([d for d in (all_cmp, k46_cmp, mixed) if not d.empty]).to_csv(
        "experiment_comparison.csv", index=False)
    print("\n(wrote experiment_comparison.csv)")

    print("\nCaveats:")
    print(" - Every K46-labelled row in this dataset is a multi-grade layered / functionally graded panel from a")
    print("   single paper, so there is no monolithic epoxy+K46 modulus-vs-vf curve here to validate against;")
    print("   the model curve above is a prediction, not a fit.")
    print(" - Measured syntactic-foam moduli typically sit 20-40% below HP-MT (matrix porosity, imperfect")
    print("   interfacial bonding, particle breakage); larger gaps here also reflect the grade mismatch.")
    print(" - No constant was tuned and no experimental value was hard-coded to close that gap.")


if __name__ == "__main__":
    main()
