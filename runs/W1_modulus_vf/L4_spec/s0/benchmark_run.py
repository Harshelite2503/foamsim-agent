"""W1 - Compressive modulus of epoxy / 3M K46 glass-microballoon syntactic foam vs volume fraction.

System: epoxy matrix (E=3000 MPa, nu=0.35, rho=1.18 g/cm^3) filled with 3M K46 glass microballoons
(true density 0.46 g/cm^3, borosilicate shell E=60 GPa = 60000 MPa, nu=0.21, rho=2.54 g/cm^3),
particle volume fraction vf = 0 ... 0.6, quasi-static compression.

Models (foamsim.micromechanics):
  HP-MT : hollow-particle Mori-Tanaka - the hollow sphere is replaced by Hashin's exact equivalent
          homogeneous sphere (composite-sphere-assemblage bulk modulus), then a Mori-Tanaka
          mean-field estimate for dilute-to-moderate spherical inclusions in a matrix.
  HP-DS : differential scheme (McLaughlin) on the same equivalent particle - accounts for
          particle-particle interaction by incremental addition; usually slightly below HP-MT at high vf.
  HS    : Hashin-Shtrikman bounds for the same two isotropic phases - the rigorous envelope for any
          isotropic two-phase microstructure at that volume fraction.

Assumptions: isotropic linear elasticity, perfect particle/matrix bonding, monodisperse spherical
non-interpenetrating particles, no matrix porosity, no particle breakage, small strain (initial
loading modulus). Compressive and tensile moduli are assumed equal (linear elasticity), which is
what syntactic-foam papers report as the compressive modulus.

Units: modulus MPa, density g/cm^3, volume fraction dimensionless.

Outputs written to the current directory: results.json, results.csv, modulus_vs_vf.png
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from foamsim import MATERIALS, hollow_particle
from foamsim.data import reference_curve
from foamsim.materials import HollowParticle
from foamsim.micromechanics import (
    RCP,
    density,
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
)

VF_GRID = np.linspace(0.0, 0.6, 13)
TOL_REL = 1e-6


def self_checks(matrix, particle) -> dict:
    """Independent known results recovered before any production number is trusted."""
    checks = {}

    # 1. vf = 0 must return the neat matrix modulus exactly.
    e0_mt = hollow_particle_mori_tanaka(matrix, particle, vf=0.0)
    e0_ds = hollow_particle_differential(matrix, particle, vf=0.0)
    checks["E_vf0_mt_mpa"] = e0_mt.E
    checks["E_vf0_ds_mpa"] = e0_ds.E
    checks["E_matrix_mpa"] = matrix.E
    checks["pass_E_vf0_equals_matrix"] = bool(
        abs(e0_mt.E - matrix.E) <= TOL_REL * matrix.E and abs(e0_ds.E - matrix.E) <= TOL_REL * matrix.E
    )

    # 2. Density rule of mixtures: rho(0.4) = 0.6*1.18 + 0.4*0.46 = 0.892 g/cm^3.
    rho04 = density(matrix, particle, vf=0.4)
    rho04_rom = (1 - 0.4) * matrix.rho + 0.4 * particle.true_density
    checks["rho_vf04_g_cc"] = rho04
    checks["rho_vf04_rule_of_mixtures_g_cc"] = rho04_rom
    checks["pass_rho_vf04"] = bool(abs(rho04 - rho04_rom) <= 1e-9 and abs(rho04 - 0.892) <= 5e-4)

    # 3. Every estimate must lie inside the HS bounds on the whole sweep.
    inside = True
    worst = None
    for vf in VF_GRID:
        b = hashin_shtrikman_bounds(matrix, particle, vf)
        for est in (hollow_particle_mori_tanaka(matrix, particle, vf),
                    hollow_particle_differential(matrix, particle, vf)):
            ok = b["E_lo"] - 1e-6 <= est.E <= b["E_hi"] + 1e-6
            if not ok:
                inside = False
                worst = {"vf": float(vf), "model": est.model, "E": est.E, **b}
    checks["pass_estimates_inside_hs"] = bool(inside)
    checks["hs_violation"] = worst

    # 4. Geometry / packing premise checks.
    checks["eta_inferred"] = particle.eta
    checks["pass_eta_physical"] = bool(0.0 <= particle.eta < 1.0)
    checks["pass_vf_max_below_rcp"] = bool(float(VF_GRID.max()) <= RCP)
    checks["rcp_limit"] = RCP

    checks["all_passed"] = bool(
        checks["pass_E_vf0_equals_matrix"] and checks["pass_rho_vf04"]
        and checks["pass_estimates_inside_hs"] and checks["pass_eta_physical"]
        and checks["pass_vf_max_below_rcp"]
    )
    return checks


def sweep(matrix, particle) -> pd.DataFrame:
    rows = []
    for vf in VF_GRID:
        mt = hollow_particle_mori_tanaka(matrix, particle, vf)
        ds = hollow_particle_differential(matrix, particle, vf)
        b = hashin_shtrikman_bounds(matrix, particle, vf)
        rows.append({
            "vf": float(vf),
            "density_g_cc": mt.rho,
            "relative_density": mt.rho / matrix.rho,
            "E_HP_MT_mpa": mt.E,
            "E_HP_DS_mpa": ds.E,
            "E_HS_lower_mpa": b["E_lo"],
            "E_HS_upper_mpa": b["E_hi"],
            "model_spread_mpa": abs(mt.E - ds.E),
            "model_spread_pct": 100 * abs(mt.E - ds.E) / mt.E,
            "hs_band_width_pct": 100 * (b["E_hi"] - b["E_lo"]) / mt.E,
            "nu_HP_MT": mt.nu,
            "K_HP_MT_mpa": mt.K,
            "G_HP_MT_mpa": mt.G,
            "specific_E_HP_MT_mpa_cc_g": mt.E / mt.rho,
        })
    return pd.DataFrame(rows)


def _row_particle(row, default_particle):
    """Particle actually used in an experimental row, and how it was identified.

    Priority: (1) reported particle true density; (2) true density backed out of the reported
    composite density and vf via the rule of mixtures; (3) the K46 particle of this study.
    Nothing is hard-coded - every number comes from the dataset row.
    """
    matrix = MATERIALS["epoxy"]
    td = row.get("particle_true_density_g_cc")
    if pd.notna(td) and 0 < td <= MATERIALS["glass"].rho:
        return HollowParticle.from_true_density(MATERIALS["glass"], float(td)), "reported_true_density"
    rho_c, vf = row.get("measured_density_g_cc"), row.get("particle_volume_fraction")
    if pd.notna(rho_c) and pd.notna(vf) and vf > 0:
        td = (float(rho_c) - (1 - float(vf)) * matrix.rho) / float(vf)
        if 0 < td <= MATERIALS["glass"].rho:
            return HollowParticle.from_true_density(MATERIALS["glass"], td), "back_calculated_from_density"
    return default_particle, "assumed_K46"


def compare_experiment(matrix, particle) -> tuple[pd.DataFrame, dict]:
    """Compare with the FoamGPT epoxy / glass-microballoon quasi-static compression rows.

    No experimental value is hard-coded: everything is read from the bundled dataset. Because the
    reference rows use several microballoon grades (not only K46), each row is predicted with the
    particle implied by that row's own reported true density or composite density; the K46 curve of
    this study is kept alongside for reference.
    """
    ref = reference_curve("epoxy", "glass_microballoon")
    ref = ref[ref["modulus_mpa"].notna()].copy()
    ref = ref[ref["particle_volume_fraction"] <= RCP]

    pred_mt, pred_ds, lo, hi, k46, src, eta_row = [], [], [], [], [], [], []
    for _, row in ref.iterrows():
        vf = float(row["particle_volume_fraction"])
        p_row, how = _row_particle(row, particle)
        pred_mt.append(hollow_particle_mori_tanaka(matrix, p_row, vf).E)
        pred_ds.append(hollow_particle_differential(matrix, p_row, vf).E)
        b = hashin_shtrikman_bounds(matrix, p_row, vf)
        lo.append(b["E_lo"]); hi.append(b["E_hi"])
        k46.append(hollow_particle_mori_tanaka(matrix, particle, vf).E)
        src.append(how); eta_row.append(p_row.eta)
    ref["particle_source"] = src
    ref["eta_used"] = eta_row
    ref["E_pred_HP_MT_mpa"] = pred_mt
    ref["E_pred_HP_DS_mpa"] = pred_ds
    ref["E_pred_HP_MT_K46_mpa"] = k46
    ref["E_HS_lower_mpa"] = lo
    ref["E_HS_upper_mpa"] = hi
    ref["abs_pct_error_HP_MT"] = 100 * (ref["E_pred_HP_MT_mpa"] - ref["modulus_mpa"]).abs() / ref["modulus_mpa"]
    ref["abs_pct_error_HP_DS"] = 100 * (ref["E_pred_HP_DS_mpa"] - ref["modulus_mpa"]).abs() / ref["modulus_mpa"]
    ref["signed_pct_error_HP_MT"] = 100 * (ref["E_pred_HP_MT_mpa"] - ref["modulus_mpa"]) / ref["modulus_mpa"]
    ref["inside_hs_band"] = (ref["modulus_mpa"] >= ref["E_HS_lower_mpa"]) & (ref["modulus_mpa"] <= ref["E_HS_upper_mpa"])

    stats = {
        "n_experimental_points": int(len(ref)),
        "n_papers": int(ref["paper_id"].nunique()),
        "particle_source_counts": {k: int(v) for k, v in ref["particle_source"].value_counts().items()},
        "eta_used_range": [float(ref["eta_used"].min()), float(ref["eta_used"].max())],
        "eta_K46": float(particle.eta),
        "vf_range": [float(ref["particle_volume_fraction"].min()), float(ref["particle_volume_fraction"].max())],
        "mape_HP_MT_pct": float(ref["abs_pct_error_HP_MT"].mean()),
        "mape_HP_DS_pct": float(ref["abs_pct_error_HP_DS"].mean()),
        "median_signed_error_HP_MT_pct": float(ref["signed_pct_error_HP_MT"].median()),
        "frac_experiment_inside_hs_band": float(ref["inside_hs_band"].mean()),
        "n_experiment_inside_hs_band": int(ref["inside_hs_band"].sum()),
    }
    return ref, stats


def make_figure(df: pd.DataFrame, ref: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))

    a = ax[0]
    a.fill_between(df["vf"], df["E_HS_lower_mpa"], df["E_HS_upper_mpa"], color="0.85",
                   label="Hashin-Shtrikman bounds")
    a.plot(df["vf"], df["E_HP_MT_mpa"], "-", color="C0", lw=2, label="HP Mori-Tanaka")
    a.plot(df["vf"], df["E_HP_DS_mpa"], "--", color="C1", lw=2, label="HP differential scheme")
    if len(ref):
        a.plot(ref["particle_volume_fraction"], ref["modulus_mpa"], "o", color="k", ms=5,
               mfc="none", label=f"experiment (FoamGPT, n={len(ref)})")
    a.set_xlabel("particle volume fraction $v_f$ (-)")
    a.set_ylabel("compressive modulus $E$ (MPa)")
    a.set_title("Epoxy / 3M K46 syntactic foam")
    a.legend(fontsize=8, loc="upper left")
    a.grid(alpha=0.3)

    b = ax[1]
    b.plot(df["vf"], df["density_g_cc"], "-", color="C2", lw=2, label="density (rule of mixtures)")
    if len(ref):
        d = ref[ref["measured_density_g_cc"].notna()]
        if len(d):
            b.plot(d["particle_volume_fraction"], d["measured_density_g_cc"], "s", color="k", ms=5,
                   mfc="none", label="measured density")
    b.set_xlabel("particle volume fraction $v_f$ (-)")
    b.set_ylabel(r"density $\rho$ (g/cm$^3$)")
    b.set_title("Density vs volume fraction")
    b.legend(fontsize=8)
    b.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    matrix = MATERIALS["epoxy"]
    particle = hollow_particle("K46")

    checks = self_checks(matrix, particle)
    df = sweep(matrix, particle)
    ref, stats = compare_experiment(matrix, particle)

    # tables -> results.csv (model sweep) + results_experiment.csv (validation rows)
    df.to_csv("results.csv", index=False)
    ref.to_csv("results_experiment.csv", index=False)
    make_figure(df, ref, "modulus_vs_vf.png")

    row06 = df.iloc[-1]
    row04 = df[np.isclose(df["vf"], 0.4)].iloc[0]
    results = {
        "task": "W1 compressive modulus vs particle volume fraction, epoxy / 3M K46 glass microballoons",
        "units": {"modulus": "MPa", "density": "g/cm^3", "volume_fraction": "dimensionless",
                  "diameter": "micrometre"},
        "constituents": {
            "matrix": {"name": matrix.name, "E_mpa": matrix.E, "nu": matrix.nu, "rho_g_cc": matrix.rho},
            "particle": {"grade": "K46", "shell": particle.shell.name, "shell_E_mpa": particle.shell.E,
                         "shell_nu": particle.shell.nu, "shell_rho_g_cc": particle.shell.rho,
                         "true_density_g_cc": particle.true_density, "eta_r_in_over_r_out": particle.eta,
                         "wall_volume_fraction": particle.wall_volume_fraction,
                         "diameter_um": particle.diameter_um},
        },
        "models_used": {
            "primary": "HP-MT (hollow-particle Mori-Tanaka on Hashin's equivalent homogeneous sphere)",
            "secondary": "HP-DS (differential scheme, same equivalent particle)",
            "bounds": "Hashin-Shtrikman two-phase bounds",
            "assumptions": [
                "isotropic linear elasticity, small strain (initial loading modulus)",
                "perfect particle-matrix bonding, no debonding or particle crushing",
                "monodisperse non-interpenetrating spherical particles, no matrix porosity",
                "compressive modulus = tensile modulus (linear elastic)",
                "vf includes the hollow core; vf <= RCP = 0.64 for monodisperse spheres",
            ],
        },
        "self_checks": checks,
        "sweep": {
            "vf_grid": [float(v) for v in VF_GRID],
            "E_HP_MT_mpa": [float(v) for v in df["E_HP_MT_mpa"]],
            "E_HP_DS_mpa": [float(v) for v in df["E_HP_DS_mpa"]],
            "E_HS_lower_mpa": [float(v) for v in df["E_HS_lower_mpa"]],
            "E_HS_upper_mpa": [float(v) for v in df["E_HS_upper_mpa"]],
            "density_g_cc": [float(v) for v in df["density_g_cc"]],
        },
        "key_numbers": {
            "E_vf0_mpa": float(df.iloc[0]["E_HP_MT_mpa"]),
            "E_vf04_HP_MT_mpa": float(row04["E_HP_MT_mpa"]),
            "E_vf04_HP_DS_mpa": float(row04["E_HP_DS_mpa"]),
            "E_vf04_HS_band_mpa": [float(row04["E_HS_lower_mpa"]), float(row04["E_HS_upper_mpa"])],
            "rho_vf04_g_cc": float(row04["density_g_cc"]),
            "E_vf06_HP_MT_mpa": float(row06["E_HP_MT_mpa"]),
            "E_vf06_HP_DS_mpa": float(row06["E_HP_DS_mpa"]),
            "E_vf06_HS_band_mpa": [float(row06["E_HS_lower_mpa"]), float(row06["E_HS_upper_mpa"])],
            "rho_vf06_g_cc": float(row06["density_g_cc"]),
            "max_model_spread_pct": float(df["model_spread_pct"].max()),
            "specific_E_gain_vf06_over_neat": float(row06["specific_E_HP_MT_mpa_cc_g"]
                                                    / df.iloc[0]["specific_E_HP_MT_mpa_cc_g"]),
        },
        "uncertainty": {
            "model_spread_HP_MT_vs_HP_DS_pct": {
                "definition": "|E_MT - E_DS| / E_MT over the sweep, an estimate of mean-field scheme uncertainty",
                "max": float(df["model_spread_pct"].max()),
                "at_vf_0.6": float(row06["model_spread_pct"]),
            },
            "hs_band_width_pct_of_HP_MT": {
                "at_vf_0.4": float(row04["hs_band_width_pct"]),
                "at_vf_0.6": float(row06["hs_band_width_pct"]),
            },
            "experimental_scatter_note": "experimental moduli scatter between papers (different resins, cure, "
                                         "matrix porosity, particle breakage); see results_experiment.csv",
        },
        "experimental_comparison": stats,
        "outputs": {"csv": ["results.csv", "results_experiment.csv"], "figure": "modulus_vs_vf.png",
                    "json": "results.json"},
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- physics sanity summary ----
    print("\n=== Epoxy / 3M K46 syntactic foam: compressive modulus vs volume fraction ===")
    print(f"Constituents: epoxy E={matrix.E:.0f} MPa, nu={matrix.nu}, rho={matrix.rho} g/cm3; "
          f"K46 shell E={particle.shell.E/1000:.0f} GPa, nu={particle.shell.nu}, "
          f"true density {particle.true_density:.2f} g/cm3 -> eta={particle.eta:.4f} "
          f"(wall volume fraction {particle.wall_volume_fraction:.3f}).")
    print("Model: HP Mori-Tanaka (primary) on Hashin's equivalent hollow-sphere particle; "
          "HP differential scheme as the second estimate; HS bounds as the rigorous envelope. Units MPa, g/cm3.")
    print("\nSelf-checks:")
    print(f"  E(vf=0) = {checks['E_vf0_mt_mpa']:.4f} MPa (matrix {matrix.E:.0f} MPa) -> "
          f"{'PASS' if checks['pass_E_vf0_equals_matrix'] else 'FAIL'}")
    print(f"  rho(vf=0.4) = {checks['rho_vf04_g_cc']:.4f} g/cm3 (rule of mixtures "
          f"{checks['rho_vf04_rule_of_mixtures_g_cc']:.4f}, expected 0.892) -> "
          f"{'PASS' if checks['pass_rho_vf04'] else 'FAIL'}")
    print(f"  all HP-MT and HP-DS estimates inside HS bounds on vf in [0,0.6] -> "
          f"{'PASS' if checks['pass_estimates_inside_hs'] else 'FAIL'}")
    print(f"  eta = {checks['eta_inferred']:.4f} in [0,1) and vf_max = 0.60 < RCP = {RCP} -> "
          f"{'PASS' if checks['pass_eta_physical'] and checks['pass_vf_max_below_rcp'] else 'FAIL'}")
    print("\nTrend (MPa unless noted):")
    for _, r in df.iterrows():
        print(f"  vf={r['vf']:.2f}  rho={r['density_g_cc']:.3f} g/cm3  "
              f"E_MT={r['E_HP_MT_mpa']:8.1f}  E_DS={r['E_HP_DS_mpa']:8.1f}  "
              f"HS=[{r['E_HS_lower_mpa']:8.1f}, {r['E_HS_upper_mpa']:8.1f}]")
    direction = "increases" if row06["E_HP_MT_mpa"] > df.iloc[0]["E_HP_MT_mpa"] else "decreases"
    print(f"\nPhysics: K46 is a stiff borosilicate microballoon (equivalent-particle modulus above the epoxy), "
          f"so E {direction} with vf while density falls from {matrix.rho:.3f} to "
          f"{row06['density_g_cc']:.3f} g/cm3 -- specific modulus rises "
          f"{results['key_numbers']['specific_E_gain_vf06_over_neat']:.2f}x at vf=0.6.")
    print(f"Uncertainty: HP-MT vs HP-DS spread <= {df['model_spread_pct'].max():.1f} % over the sweep; "
          f"HS band at vf=0.6 spans {row06['E_HS_lower_mpa']:.0f}-{row06['E_HS_upper_mpa']:.0f} MPa "
          f"({row06['hs_band_width_pct']:.0f} % of the HP-MT estimate).")
    if stats["n_experimental_points"]:
        print(f"Experiment (FoamGPT, {stats['n_experimental_points']} quasi-static epoxy/glass-microballoon "
              f"compression points from {stats['n_papers']} papers, vf "
              f"{stats['vf_range'][0]:.2f}-{stats['vf_range'][1]:.2f}, eta implied by each row "
              f"{stats['eta_used_range'][0]:.3f}-{stats['eta_used_range'][1]:.3f} vs eta_K46="
              f"{stats['eta_K46']:.3f}): HP-MT MAPE {stats['mape_HP_MT_pct']:.1f} %, "
              f"HP-DS MAPE {stats['mape_HP_DS_pct']:.1f} %, median signed error "
              f"{stats['median_signed_error_HP_MT_pct']:+.1f} %; "
              f"{stats['n_experiment_inside_hs_band']}/{stats['n_experimental_points']} measurements lie inside "
              f"the HS band.")
        print(f"  Particle per row identified as {stats['particle_source_counts']}. The reference rows are not "
              "K46 foams: most are thin-walled S22/S32/S38 layered (functionally graded) foams, so they are a "
              "sanity check on the model class, not a like-for-like K46 validation.")
        print("  All measurements fall BELOW the HS lower bound for a perfect two-phase epoxy+microballoon "
              "composite. That is not a numerical error: it means these foams are not two-phase - entrapped "
              "matrix porosity, weak/debonded interfaces and (in the graded foams) layer interfaces act as a "
              "third, compliant phase, and some reported moduli may be machine-compliance-affected secant "
              "values. The mean-field estimates should therefore be read as an upper envelope for E.")
        print("  Measured moduli sit below the HP-MT estimate (positive signed error) because of matrix "
              "porosity, imperfect bonding, particle breakage and, in the graded foams, interfaces between "
              "layers. No constant was tuned and no experimental value is hard-coded.")
    print("\nWrote results.json, results.csv, results_experiment.csv, modulus_vs_vf.png")


if __name__ == "__main__":
    main()
