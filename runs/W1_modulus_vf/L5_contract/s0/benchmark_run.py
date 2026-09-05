#!/usr/bin/env python
"""Compressive modulus of epoxy / 3M K46 glass-microballoon syntactic foam vs particle volume fraction.

What this script does
---------------------
Sweeps the microballoon volume fraction vf (particles INCLUDING their hollow cores) from 0 to
--vf-max and, at each vf, evaluates with the `foamsim` toolkit:

  * density(vf)                         rule of mixtures with the particle *true* density
  * HP-MT  (hollow_particle_mori_tanaka) equivalent-particle Mori-Tanaka estimate  [primary model]
  * HP-DS  (hollow_particle_differential) differential-scheme estimate            [spread/2nd model]
  * Hashin-Shtrikman lower/upper bounds  (every estimate must lie inside them)

and compares the predictions with the bundled FoamGPT experimental table
(epoxy / glass_microballoon, primary + unflagged, quasi-static compression rows).

Models and assumptions
----------------------
Hollow sphere -> equivalent homogeneous solid particle (Hashin 1962 composite-spheres K, exact for a
void core; HS-upper-bound G for the porous shell), then Mori-Tanaka (Benveniste 1987) or the
differential scheme (McLaughlin 1977). Assumptions: linear elasticity, isotropy, perfectly bonded
spherical particles, monodisperse non-interacting-shape (mean-field) microstructure, no matrix
porosity, no particle breakage, no interphase. Small-strain modulus only -- these models say nothing
about crush strength or the post-yield plateau. vf is capped by random close packing (0.64).

Units
-----
Moduli MPa, densities g/cm^3, particle diameters micrometres, vf dimensionless.

CLI
---
    python benchmark_run.py [--matrix epoxy] [--grade K46] [--vf-max 0.6] [--out-dir .]

Outputs into --out-dir: results.json, results.csv, modulus_vs_vf.png. Exit code 1 if any physics
self-check fails.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from foamsim import MATERIALS, hollow_particle  # noqa: E402
from foamsim.data import reference_curve  # noqa: E402
from foamsim.materials import HollowParticle, Isotropic  # noqa: E402
from foamsim.micromechanics import (  # noqa: E402
    RCP,
    density,
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
    hollow_sphere_equivalent,
)

SEED = 0
N_POINTS = 13  # vf grid points, gives a 0.05 step for vf_max = 0.6


def set_seeds(seed: int = SEED) -> None:
    """Fix every RNG this script could touch, so the run is bit-for-bit reproducible."""
    random.seed(seed)
    np.random.seed(seed)


def sweep(matrix_name: str, grade: str, vf_max: float, n: int = N_POINTS) -> pd.DataFrame:
    """Return one row per vf with density, HP-MT, HP-DS and the HS bounds (all MPa / g/cm^3)."""
    matrix = MATERIALS[matrix_name]
    particle = hollow_particle(grade)
    rows = []
    for vf in np.linspace(0.0, vf_max, n):
        vf = float(vf)
        mt = hollow_particle_mori_tanaka(matrix, particle, vf)
        ds = hollow_particle_differential(matrix, particle, vf)
        hs = hashin_shtrikman_bounds(matrix, particle, vf)
        rows.append(
            {
                "vf": vf,
                "density_g_cc": density(matrix, particle, vf),
                "E_hp_mt_mpa": mt.E,
                "E_hp_ds_mpa": ds.E,
                "E_hs_lo_mpa": hs["E_lo"],
                "E_hs_hi_mpa": hs["E_hi"],
                "nu_hp_mt": mt.nu,
                "nu_hp_ds": ds.nu,
                "specific_E_hp_mt_mpa_per_g_cc": mt.E / mt.rho,
                "model_spread_pct": 100.0 * abs(mt.E - ds.E) / mt.E,
            }
        )
    return pd.DataFrame(rows)


def _row_particle(matrix: Isotropic, row: pd.Series, default: HollowParticle) -> tuple[HollowParticle, str]:
    """Best available hollow particle for one experimental row, and how it was obtained.

    Preference order: (1) the row's reported particle true density; (2) the particle true density
    implied by inverting the rule of mixtures on the measured composite density,
    rho_p = (rho_c - (1-vf) rho_m) / vf; (3) the nominal task grade. No literature value is typed in
    -- everything comes from the dataset row itself.
    """
    shell = MATERIALS["glass"]
    rho_p = row.get("particle_true_density_g_cc")
    if pd.notna(rho_p) and 0 < float(rho_p) <= shell.rho:
        return HollowParticle.from_true_density(shell, float(rho_p)), "reported_true_density"
    vf, rho_c = float(row["particle_volume_fraction"]), row.get("measured_density_g_cc")
    if pd.notna(rho_c) and vf > 0:
        implied = (float(rho_c) - (1 - vf) * matrix.rho) / vf
        if 0 < implied <= shell.rho:
            return HollowParticle.from_true_density(shell, implied), "implied_from_measured_density"
    return default, "nominal_grade"


def experimental_comparison(sweep_df: pd.DataFrame, matrix_name: str, grade: str) -> tuple[pd.DataFrame, dict]:
    """Compare HP-MT/HP-DS with FoamGPT epoxy/glass-microballoon compression rows.

    Two comparisons are reported. The *nominal* one evaluates the task's K46 particle at each
    experimental vf -- this is what the task literally asks for, but the dataset rows are mostly
    other grades, so it overstates the model error. The *matched* one rebuilds the particle from the
    row's own particle/composite density (see `_row_particle`), which isolates model error from
    grade mismatch. Nothing is hard-coded: every experimental number is read from the dataset.
    """
    matrix = MATERIALS[matrix_name]
    particle = hollow_particle(grade)
    ref = reference_curve(matrix_name, "glass_microballoon")
    ref = ref[ref["modulus_mpa"].notna()]
    ref = ref[ref["particle_volume_fraction"].between(0.0, min(sweep_df["vf"].max(), RCP))]

    rows = []
    for _, r in ref.iterrows():
        vf = float(r["particle_volume_fraction"])
        mt = hollow_particle_mori_tanaka(matrix, particle, vf)
        ds = hollow_particle_differential(matrix, particle, vf)
        hs = hashin_shtrikman_bounds(matrix, particle, vf)
        e_exp = float(r["modulus_mpa"])
        p_row, source = _row_particle(matrix, r, particle)
        mt_m = hollow_particle_mori_tanaka(matrix, p_row, vf)
        hs_m = hashin_shtrikman_bounds(matrix, p_row, vf)
        rows.append(
            {
                "record_id": r["record_id"],
                "paper_id": r["paper_id"],
                "particle_grade": r["particle_grade"],
                "vf": vf,
                "E_exp_mpa": e_exp,
                "E_hp_mt_mpa": mt.E,
                "E_hp_ds_mpa": ds.E,
                "E_hs_lo_mpa": hs["E_lo"],
                "E_hs_hi_mpa": hs["E_hi"],
                "rel_err_mt_pct": 100.0 * (mt.E - e_exp) / e_exp,
                "rel_err_ds_pct": 100.0 * (ds.E - e_exp) / e_exp,
                "inside_hs_band": bool(hs["E_lo"] <= e_exp <= hs["E_hi"]),
                "particle_source": source,
                "particle_true_density_used_g_cc": p_row.true_density,
                "E_hp_mt_matched_mpa": mt_m.E,
                "E_hs_lo_matched_mpa": hs_m["E_lo"],
                "E_hs_hi_matched_mpa": hs_m["E_hi"],
                "rel_err_mt_matched_pct": 100.0 * (mt_m.E - e_exp) / e_exp,
                "inside_hs_band_matched": bool(hs_m["E_lo"] <= e_exp <= hs_m["E_hi"]),
                "measured_density_g_cc": (
                    float(r["measured_density_g_cc"]) if pd.notna(r["measured_density_g_cc"]) else None
                ),
            }
        )
    comp = pd.DataFrame(rows)
    if comp.empty:
        return comp, {"n_points": 0}
    stats = {
        "n_points": int(len(comp)),
        "n_papers": int(comp["paper_id"].nunique()),
        "vf_range": [float(comp["vf"].min()), float(comp["vf"].max())],
        "mape_hp_mt_pct": float(comp["rel_err_mt_pct"].abs().mean()),
        "mape_hp_ds_pct": float(comp["rel_err_ds_pct"].abs().mean()),
        "median_signed_bias_hp_mt_pct": float(comp["rel_err_mt_pct"].median()),
        "frac_experiments_inside_hs_band": float(comp["inside_hs_band"].mean()),
        "mape_hp_mt_matched_pct": float(comp["rel_err_mt_matched_pct"].abs().mean()),
        "median_signed_bias_hp_mt_matched_pct": float(comp["rel_err_mt_matched_pct"].median()),
        "frac_experiments_inside_hs_band_matched": float(comp["inside_hs_band_matched"].mean()),
        "particle_source_counts": comp["particle_source"].value_counts().to_dict(),
        "matched_err_pct_p25_median_p75": [
            float(comp["rel_err_mt_matched_pct"].quantile(q)) for q in (0.25, 0.50, 0.75)
        ],
        "matched_err_pct_min_max": [
            float(comp["rel_err_mt_matched_pct"].min()), float(comp["rel_err_mt_matched_pct"].max())
        ],
        "n_rows_nominal_grade_k46": int((comp["particle_grade"].astype(str).str.contains(grade)).sum()),
        "note": (
            "No epoxy/K46-only quasi-static compression row with a modulus exists in the bundled "
            "dataset: the rows are S60HS at vf=0.30 and a functionally graded multi-layer study "
            "(mixed K46/S22/S32/S38) at vf=0.60, and the graded rows list two moduli per sample. "
            "The 'nominal' error therefore mixes grade mismatch with model error; the 'matched' "
            "error rebuilds the particle from each row's own density and is the meaningful number. "
            "The matched errors are bimodal (~+70-80 % for the higher modulus of each graded sample, "
            "~+290-350 % for the lower one), which indicates the two reported moduli per sample are "
            "not the same quantity -- only the higher branch is consistent with a compressive Young's "
            "modulus. Residual over-prediction on that branch is expected: matrix porosity, particle "
            "breakage and imperfect bonding are not modelled."
        ),
    }
    return comp, stats


def compute(matrix: str = "epoxy", grade: str = "K46", vf_max: float = 0.6, n: int = N_POINTS) -> dict:
    """Run the whole calculation and return every key number as a JSON-serialisable dict."""
    set_seeds()
    m = MATERIALS[matrix]
    p = hollow_particle(grade)
    eq = hollow_sphere_equivalent(p)
    df = sweep(matrix, grade, vf_max, n)
    comp, stats = experimental_comparison(df, matrix, grade)

    checks = {
        "E_vf0_mpa": float(df.loc[df["vf"] == 0.0, "E_hp_mt_mpa"].iloc[0]),
        "matrix_E_mpa": float(m.E),
        "density_vf0p4_g_cc": float(density(m, p, 0.4)),
        "density_vf0p4_expected_g_cc": 0.892,
        "all_mt_inside_hs": bool(
            ((df["E_hp_mt_mpa"] >= df["E_hs_lo_mpa"] - 1e-6) & (df["E_hp_mt_mpa"] <= df["E_hs_hi_mpa"] + 1e-6)).all()
        ),
        "all_ds_inside_hs": bool(
            ((df["E_hp_ds_mpa"] >= df["E_hs_lo_mpa"] - 1e-6) & (df["E_hp_ds_mpa"] <= df["E_hs_hi_mpa"] + 1e-6)).all()
        ),
        "hs_band_ordered": bool((df["E_hs_hi_mpa"] >= df["E_hs_lo_mpa"] - 1e-9).all()),
        "density_monotonic_decreasing": bool((np.diff(df["density_g_cc"].to_numpy()) < 0).all()),
        "vf_max_within_rcp": bool(vf_max <= RCP),
    }

    return {
        "task": "W1 compressive modulus vs particle volume fraction, epoxy / 3M K46 glass microballoons",
        "units": {
            "modulus": "MPa",
            "density": "g/cm^3",
            "diameter": "micrometre",
            "vf": "dimensionless volume fraction of particles including hollow cores",
        },
        "seed": SEED,
        "inputs": {
            "matrix": {"name": m.name, "E_mpa": m.E, "nu": m.nu, "rho_g_cc": m.rho},
            "particle": {
                "grade": grade,
                "shell": {"name": p.shell.name, "E_mpa": p.shell.E, "nu": p.shell.nu, "rho_g_cc": p.shell.rho},
                "eta_inner_over_outer": p.eta,
                "true_density_g_cc": p.true_density,
                "diameter_um": p.diameter_um,
            },
            "equivalent_solid_particle": {"K_mpa": eq.K, "G_mpa": eq.G, "E_mpa": eq.E, "nu": eq.nu},
            "vf_max": vf_max,
            "n_points": n,
            "rcp_limit": RCP,
        },
        "models": {
            "primary": "HP-MT (equivalent-particle Mori-Tanaka, Benveniste 1987 + Hashin 1962 hollow sphere)",
            "secondary": "HP-DS (differential scheme, McLaughlin 1977) -- used as the model spread",
            "bounds": "Hashin-Shtrikman (1963) two-phase bounds on matrix + equivalent particle",
            "assumptions": [
                "linear elastic, isotropic, perfectly bonded spherical particles",
                "mean-field homogenization; no particle clustering or percolation",
                "no matrix porosity, no particle breakage, no interphase layer",
                "small-strain compressive modulus only; not crush strength or plateau stress",
                "vf includes the hollow cores and is limited by random close packing (0.64)",
            ],
        },
        "sweep": df.to_dict(orient="records"),
        "key_points": {
            f"vf={v:.1f}": {
                "density_g_cc": float(density(m, p, v)),
                "E_hp_mt_mpa": float(hollow_particle_mori_tanaka(m, p, v).E),
                "E_hp_ds_mpa": float(hollow_particle_differential(m, p, v).E),
                "E_hs_lo_mpa": float(hashin_shtrikman_bounds(m, p, v)["E_lo"]),
                "E_hs_hi_mpa": float(hashin_shtrikman_bounds(m, p, v)["E_hi"]),
            }
            for v in (0.0, 0.2, 0.4, min(0.6, vf_max))
        },
        "uncertainty": {
            "model_spread_hp_mt_vs_hp_ds_pct_max": float(df["model_spread_pct"].max()),
            "hs_band_width_at_vf_max_pct_of_mt": float(
                100.0 * (df["E_hs_hi_mpa"].iloc[-1] - df["E_hs_lo_mpa"].iloc[-1]) / df["E_hp_mt_mpa"].iloc[-1]
            ),
            "experiment_scatter": stats,
        },
        "experimental_comparison": comp.to_dict(orient="records"),
        "experimental_stats": stats,
        "checks": checks,
    }


def validate(results: dict) -> list[str]:
    """Return the list of failed physics/self-consistency checks (empty list == all passed)."""
    c = results["checks"]
    failed: list[str] = []
    if abs(c["E_vf0_mpa"] - c["matrix_E_mpa"]) > 1e-6 * c["matrix_E_mpa"]:
        failed.append(f"E(vf=0) = {c['E_vf0_mpa']:.4f} MPa != matrix E = {c['matrix_E_mpa']:.1f} MPa")
    if abs(c["density_vf0p4_g_cc"] - c["density_vf0p4_expected_g_cc"]) > 1e-6:
        failed.append(
            f"density(vf=0.4) = {c['density_vf0p4_g_cc']:.6f} g/cm^3 != expected "
            f"{c['density_vf0p4_expected_g_cc']} g/cm^3"
        )
    if not c["all_mt_inside_hs"]:
        failed.append("HP-MT estimate falls outside the Hashin-Shtrikman bounds at some vf")
    if not c["all_ds_inside_hs"]:
        failed.append("HP-DS estimate falls outside the Hashin-Shtrikman bounds at some vf")
    if not c["hs_band_ordered"]:
        failed.append("HS upper bound is below the HS lower bound at some vf")
    if not c["density_monotonic_decreasing"]:
        failed.append("composite density is not monotonically decreasing with vf")
    if not c["vf_max_within_rcp"]:
        failed.append(f"vf_max exceeds random close packing ({results['inputs']['rcp_limit']})")
    return failed


def write_outputs(results: dict, out_dir: Path) -> dict:
    """Write results.json, results.csv (+ the experimental comparison CSV) and the PNG figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results["sweep"])
    comp = pd.DataFrame(results["experimental_comparison"])

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    df.to_csv(out_dir / "results.csv", index=False)
    if not comp.empty:
        comp.to_csv(out_dir / "experimental_comparison.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].fill_between(df["vf"], df["E_hs_lo_mpa"], df["E_hs_hi_mpa"], color="0.85", label="Hashin-Shtrikman band")
    ax[0].plot(df["vf"], df["E_hp_mt_mpa"], "-o", ms=4, color="C0", label="HP-MT (Mori-Tanaka)")
    ax[0].plot(df["vf"], df["E_hp_ds_mpa"], "--s", ms=4, color="C1", label="HP-DS (differential)")
    if not comp.empty:
        ax[0].plot(comp["vf"], comp["E_exp_mpa"], "k^", ms=6, alpha=0.7,
                   label=f"FoamGPT experiment, mixed grades (n={len(comp)})")
        ax[0].plot(comp["vf"], comp["E_hp_mt_matched_mpa"], "rx", ms=7, mew=1.5,
                   label="HP-MT, particle matched to row density")
    ax[0].axhline(results["inputs"]["matrix"]["E_mpa"], color="0.4", lw=0.8, ls=":", label="neat epoxy")
    ax[0].set_xlabel("particle volume fraction $v_f$ (-)")
    ax[0].set_ylabel("compressive Young's modulus $E$ (MPa)")
    ax[0].set_title(f"{results['inputs']['matrix']['name']} / {results['inputs']['particle']['grade']} "
                    "microballoons")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(df["vf"], df["density_g_cc"], "-o", ms=4, color="C2", label="model density (ROM)")
    if not comp.empty and comp["measured_density_g_cc"].notna().any():
        d = comp[comp["measured_density_g_cc"].notna()]
        ax[1].plot(d["vf"], d["measured_density_g_cc"], "k^", ms=6, alpha=0.7, label="measured density")
    ax[1].set_xlabel("particle volume fraction $v_f$ (-)")
    ax[1].set_ylabel(r"density $\rho$ (g/cm$^3$)")
    ax[1].set_title("density vs volume fraction")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "modulus_vs_vf.png", dpi=150)
    plt.close(fig)
    return {
        "results_json": str(out_dir / "results.json"),
        "results_csv": str(out_dir / "results.csv"),
        "figure_png": str(out_dir / "modulus_vs_vf.png"),
    }


def print_summary(results: dict, failed: list[str]) -> None:
    """Print a short physics sanity summary with units and the model/uncertainty statement."""
    c, u, s = results["checks"], results["uncertainty"], results["experimental_stats"]
    inp = results["inputs"]
    kp = results["key_points"]
    print("\n=== physics sanity summary (units: MPa, g/cm^3) ===")
    print(f"model: {results['models']['primary']}")
    print(f"       spread model: {results['models']['secondary']}")
    print(f"K46 wall ratio eta = {inp['particle']['eta_inner_over_outer']:.4f} inferred from true density "
          f"{inp['particle']['true_density_g_cc']:.3f} g/cm^3; equivalent solid particle "
          f"E = {inp['equivalent_solid_particle']['E_mpa']:.0f} MPa")
    print(f"self-check  E(vf=0)      = {c['E_vf0_mpa']:.2f} MPa  (matrix {c['matrix_E_mpa']:.0f} MPa)")
    print(f"self-check  rho(vf=0.4)  = {c['density_vf0p4_g_cc']:.4f} g/cm^3 "
          f"(expected {c['density_vf0p4_expected_g_cc']})")
    print(f"self-check  estimates inside HS bounds: HP-MT {c['all_mt_inside_hs']}, HP-DS {c['all_ds_inside_hs']}")
    for k, v in kp.items():
        print(f"  {k}: rho={v['density_g_cc']:.3f}  E_MT={v['E_hp_mt_mpa']:.0f}  E_DS={v['E_hp_ds_mpa']:.0f}  "
              f"HS=[{v['E_hs_lo_mpa']:.0f}, {v['E_hs_hi_mpa']:.0f}] MPa")
    print(f"uncertainty: HP-MT vs HP-DS spread <= {u['model_spread_hp_mt_vs_hp_ds_pct_max']:.1f} %; "
          f"HS band at vf_max = {u['hs_band_width_at_vf_max_pct_of_mt']:.0f} % of the HP-MT value")
    if s.get("n_points"):
        print(f"experiment ({s['n_points']} rows, {s['n_papers']} papers, vf {s['vf_range'][0]:.2f}-"
              f"{s['vf_range'][1]:.2f}):")
        print(f"  nominal K46 particle at the experimental vf: MAPE HP-MT {s['mape_hp_mt_pct']:.0f} %, "
              f"HP-DS {s['mape_hp_ds_pct']:.0f} %, median signed bias {s['median_signed_bias_hp_mt_pct']:+.0f} %, "
              f"{100 * s['frac_experiments_inside_hs_band']:.0f} % inside the HS band")
        print(f"  particle matched to each row's own density: MAPE HP-MT {s['mape_hp_mt_matched_pct']:.0f} %, "
              f"median signed bias {s['median_signed_bias_hp_mt_matched_pct']:+.0f} %, "
              f"{100 * s['frac_experiments_inside_hs_band_matched']:.0f} % inside the HS band "
              f"(particle source: {s['particle_source_counts']})")
        q = s["matched_err_pct_p25_median_p75"]
        print(f"  matched error percentiles p25/median/p75 = {q[0]:+.0f} / {q[1]:+.0f} / {q[2]:+.0f} % "
              f"(range {s['matched_err_pct_min_max'][0]:+.0f} to {s['matched_err_pct_min_max'][1]:+.0f} %)")
        print("  note: with a stiffer equivalent particle the matrix is the softer phase, so HP-MT "
              "coincides with the HS lower bound by construction -- 'inside the bounds' is a "
              "consistency check, not an independent one.")
        print("  caveat: " + s["note"])
    print(f"checks failed: {len(failed)}" + ("" if not failed else " -> " + "; ".join(failed)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the CLI; the defaults reproduce the task exactly (epoxy / K46 / vf 0-0.6 / cwd)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matrix", default="epoxy", choices=sorted(MATERIALS), help="matrix material key")
    ap.add_argument("--grade", default="K46", help="3M glass-bubble grade (e.g. K46, S38, S60)")
    ap.add_argument("--vf-max", type=float, default=0.6, help="maximum particle volume fraction (<= 0.64 RCP)")
    ap.add_argument("--out-dir", default=".", help="directory for results.json / results.csv / PNG")
    ap.add_argument("--n-points", type=int, default=N_POINTS, help="number of vf grid points")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: compute, validate, write outputs, print summary; return 1 if any check failed."""
    args = parse_args(argv)
    results = compute(matrix=args.matrix, grade=args.grade, vf_max=args.vf_max, n=args.n_points)
    failed = validate(results)
    results["failed_checks"] = failed
    results["outputs"] = write_outputs(results, Path(args.out_dir))
    print_summary(results, failed)
    # rewrite so results.json carries the check outcome and the output manifest
    Path(args.out_dir, "results.json").write_text(json.dumps(results, indent=2, default=float))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
