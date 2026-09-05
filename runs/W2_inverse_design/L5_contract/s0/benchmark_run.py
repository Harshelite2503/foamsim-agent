#!/usr/bin/env python3
"""Inverse design of the lightest epoxy / glass-microballoon syntactic foam meeting E >= 3500 MPa.

Problem
-------
Minimise composite density rho(eta, vf) subject to the compressive Young's modulus
E(eta, vf) >= E_target, over the hollow-particle wall ratio eta = r_inner/r_outer and the
particle volume fraction vf (particles counted INCLUDING their hollow cores).

Model
-----
HP-MT: hollow-particle Mori-Tanaka (`foamsim.micromechanics.hollow_particle_mori_tanaka`).
Each hollow glass sphere is first replaced by an equivalent homogeneous solid particle
(Hashin 1962 exact composite-sphere K for a void core; HS upper bound for G of the porous
shell), which is then embedded in the epoxy matrix with the Mori-Tanaka / Benveniste (1987)
scheme -- algebraically the Hashin-Shtrikman estimate with the matrix as reference medium.
HP-DS (differential scheme, McLaughlin 1977) is evaluated alongside it as a model-spread
estimate; the Hashin-Shtrikman bounds are reported as the rigorous feasibility envelope.

Assumptions
-----------
* Linear elasticity, isotropy, perfect matrix/particle bonding, no particle breakage.
* Monodisperse, randomly dispersed, non-interacting-in-a-mean-field sense spheres.
* Fully dense matrix (matrix_porosity = 0); no voids beyond the microballoon cores.
* Small strain: the "compressive modulus" is the initial elastic slope, not a crush plateau.
* Real syntactic foams typically measure 20-40 % below HP-MT (matrix porosity, broken
  balloons, imperfect interface), so the reported optimum is a model optimum, not a
  measured one.

Units
-----
Moduli in MPa, densities in g/cm^3, particle diameters in micrometres; eta, vf, nu are
dimensionless.

Outputs (written to --out-dir): results.json, results.csv, tradeoff.png.
Exit code 1 if any validation check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from foamsim import MATERIALS  # noqa: E402
from foamsim.materials import HollowParticle, hollow_particle  # noqa: E402
from foamsim.micromechanics import (  # noqa: E402
    RCP,
    density,
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
)

SEED = 0
E_TARGET_MPA = 3500.0
ETA_MIN, ETA_MAX = 0.80, 0.97
GRADES = ["K1", "K15", "K20", "K25", "S22", "S32", "S38", "K46", "S60"]


def set_seeds(seed: int = SEED) -> None:
    """Fix all RNG seeds so the run is bit-for-bit reproducible (no stochastic step is used)."""
    random.seed(seed)
    np.random.seed(seed)


def make_particle(eta: float, shell_name: str = "glass") -> HollowParticle:
    """Build a hollow glass microballoon of wall ratio `eta` (r_inner / r_outer)."""
    return HollowParticle(MATERIALS[shell_name], eta=float(eta))


def eta_of_grade(grade: str) -> float:
    """Wall ratio eta implied by a 3M glass-bubble grade's datasheet true density."""
    return hollow_particle(grade).eta


def evaluate(matrix, eta: float, vf: float) -> dict[str, Any]:
    """Evaluate one (eta, vf) design point.

    Returns HP-MT and HP-DS moduli, density, Poisson ratio, specific modulus and the
    Hashin-Shtrikman bounds at that composition. All moduli MPa, density g/cm^3.
    """
    p = make_particle(eta)
    mt = hollow_particle_mori_tanaka(matrix, p, vf)
    ds = hollow_particle_differential(matrix, p, vf)
    hs = hashin_shtrikman_bounds(matrix, p, vf)
    return {
        "eta": float(eta),
        "vf": float(vf),
        "particle_true_density_g_cc": p.true_density,
        "E_mt_mpa": mt.E,
        "E_ds_mpa": ds.E,
        "nu_mt": mt.nu,
        "rho_g_cc": mt.rho,
        "specific_E_mpa_cc_g": mt.E / mt.rho,
        "E_hs_lo_mpa": hs["E_lo"],
        "E_hs_hi_mpa": hs["E_hi"],
        "inside_hs_band": bool(hs["E_lo"] - 1e-6 <= mt.E <= hs["E_hi"] + 1e-6),
    }


def self_check(matrix) -> dict[str, Any]:
    """Independent known-result recovery: at vf = 0 the composite must be the neat matrix.

    3000 MPa is below the 3500 MPa target, i.e. the unreinforced epoxy is infeasible and the
    optimiser genuinely has to add stiff microballoons.
    """
    p = make_particle(0.90)
    e0 = hollow_particle_mori_tanaka(matrix, p, 0.0)
    return {
        "E_at_vf0_mpa": e0.E,
        "expected_matrix_E_mpa": matrix.E,
        "rho_at_vf0_g_cc": e0.rho,
        "expected_matrix_rho_g_cc": matrix.rho,
        "matrix_alone_meets_target": bool(e0.E >= E_TARGET_MPA),
    }


def optimise(matrix, eta_grid: np.ndarray, vf_grid: np.ndarray, e_target: float) -> dict[str, Any]:
    """Grid search over (eta, vf) for the minimum-density design with E_HP-MT >= e_target.

    Ties in density are broken by the larger modulus margin. Returns the optimum record,
    the full feasible/infeasible grid, and the density-vs-modulus Pareto front.
    """
    rows = [evaluate(matrix, eta, vf) for eta in eta_grid for vf in vf_grid]
    for r in rows:
        r["feasible"] = bool(r["E_mt_mpa"] >= e_target)
        r["hs_feasible"] = bool(r["E_hs_hi_mpa"] >= e_target)
    feas = [r for r in rows if r["feasible"]]
    best = min(feas, key=lambda r: (r["rho_g_cc"], -r["E_mt_mpa"])) if feas else None
    return {"rows": rows, "best": best}


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Density-vs-modulus trade-off front: for each achievable modulus level, the lightest design.

    A point is kept if no other design is both lighter and at least as stiff.
    """
    out = []
    for r in rows:
        dominated = any(
            (o["rho_g_cc"] < r["rho_g_cc"] - 1e-12 and o["E_mt_mpa"] >= r["E_mt_mpa"] - 1e-9)
            for o in rows
        )
        if not dominated:
            out.append(r)
    return sorted(out, key=lambda r: r["rho_g_cc"])


def grade_table(matrix, vf_grid: np.ndarray, e_target: float) -> list[dict[str, Any]]:
    """Best (lightest feasible) vf for each named 3M grade, for a catalogue-realisable answer."""
    out = []
    for g in GRADES:
        eta = eta_of_grade(g)
        cand = [evaluate(matrix, eta, vf) for vf in vf_grid]
        feas = [c for c in cand if c["E_mt_mpa"] >= e_target]
        rec = {"grade": g, "eta": eta, "eta_in_requested_range": bool(ETA_MIN <= eta <= ETA_MAX)}
        if feas:
            b = min(feas, key=lambda r: r["rho_g_cc"])
            rec.update({"feasible": True, "vf": b["vf"], "rho_g_cc": b["rho_g_cc"],
                        "E_mt_mpa": b["E_mt_mpa"], "E_ds_mpa": b["E_ds_mpa"],
                        "E_hs_hi_mpa": b["E_hs_hi_mpa"]})
        else:
            b = max(cand, key=lambda r: r["E_mt_mpa"])
            rec.update({"feasible": False, "vf": b["vf"], "rho_g_cc": b["rho_g_cc"],
                        "E_mt_mpa": b["E_mt_mpa"], "E_ds_mpa": b["E_ds_mpa"],
                        "E_hs_hi_mpa": b["E_hs_hi_mpa"]})
        out.append(rec)
    return out


def compute(matrix_name: str = "epoxy", grade: str | None = None, vf_max: float = 0.60,
            e_target: float = E_TARGET_MPA, n_eta: int = 69, n_vf: int = 61,
            seed: int = SEED) -> dict:
    """Run the whole inverse-design study and return every key number as a dict.

    Parameters
    ----------
    matrix_name : key into foamsim.MATERIALS (default "epoxy": E=3000 MPa, nu=0.35, rho=1.18).
    grade       : optional 3M grade; if given, eta is fixed to that grade's datasheet value and
                  only vf is optimised. If None, eta is swept continuously over [0.80, 0.97].
    vf_max      : upper bound on particle volume fraction (must be <= RCP = 0.64).
    e_target    : required compressive Young's modulus, MPa.
    """
    set_seeds(seed)
    matrix = MATERIALS[matrix_name]
    if vf_max > RCP:
        raise ValueError(f"vf_max={vf_max} exceeds random close packing {RCP}")

    vf_grid = np.linspace(0.0, vf_max, n_vf)
    if grade is None:
        eta_grid = np.linspace(ETA_MIN, ETA_MAX, n_eta)
    else:
        eta_grid = np.array([eta_of_grade(grade)])

    check = self_check(matrix)
    opt = optimise(matrix, eta_grid, vf_grid, e_target)
    best = opt["best"]
    rows = opt["rows"]

    # Rigorous feasibility: is the target below the HS upper bound anywhere on the grid?
    hs_max = max(r["E_hs_hi_mpa"] for r in rows)
    hs_at_best = None
    if best is not None:
        hs_at_best = {"E_lo": best["E_hs_lo_mpa"], "E_hi": best["E_hs_hi_mpa"],
                      "target_within_bounds": bool(best["E_hs_lo_mpa"] <= e_target <= best["E_hs_hi_mpa"]),
                      "estimate_inside_band": best["inside_hs_band"]}

    front = pareto_front(rows)
    grades = grade_table(matrix, vf_grid, e_target)
    best_grade = min([g for g in grades if g["feasible"]], key=lambda g: g["rho_g_cc"], default=None)

    spread = None
    if best is not None:
        spread = {
            "E_mt_mpa": best["E_mt_mpa"],
            "E_ds_mpa": best["E_ds_mpa"],
            "abs_diff_mpa": abs(best["E_mt_mpa"] - best["E_ds_mpa"]),
            "rel_diff_pct": 100 * abs(best["E_mt_mpa"] - best["E_ds_mpa"]) / best["E_mt_mpa"],
            "ds_meets_target": bool(best["E_ds_mpa"] >= e_target),
            "hs_band_width_pct": 100 * (best["E_hs_hi_mpa"] - best["E_hs_lo_mpa"]) / best["E_mt_mpa"],
            "note": ("model spread = HP-MT vs HP-DS; experimental syntactic foams typically fall "
                     "20-40 % below HP-MT (matrix porosity, balloon breakage)."),
        }

    return {
        "units": {"modulus": "MPa", "density": "g/cm^3", "diameter": "um",
                  "eta": "dimensionless (r_inner/r_outer)", "vf": "dimensionless"},
        "model": {
            "primary": "HP-MT (hollow-particle Mori-Tanaka via Hashin equivalent solid sphere)",
            "cross_check": "HP-DS (differential scheme, McLaughlin 1977)",
            "bounds": "Hashin-Shtrikman two-phase bounds (matrix + equivalent particle)",
            "assumptions": [
                "linear isotropic elasticity, perfect bonding, no particle breakage",
                "monodisperse randomly dispersed hollow spheres",
                "fully dense matrix (matrix_porosity = 0)",
                "initial elastic modulus, not the post-crush plateau",
            ],
        },
        "inputs": {"matrix": matrix_name, "matrix_E_mpa": matrix.E, "matrix_nu": matrix.nu,
                   "matrix_rho_g_cc": matrix.rho, "shell": "glass", "shell_E_mpa": MATERIALS["glass"].E,
                   "shell_rho_g_cc": MATERIALS["glass"].rho, "E_target_mpa": e_target,
                   "eta_range": [ETA_MIN, ETA_MAX] if grade is None else [float(eta_grid[0])] * 2,
                   "grade_fixed": grade, "vf_max": vf_max, "rcp_limit": RCP,
                   "n_eta": int(len(eta_grid)), "n_vf": int(n_vf), "seed": seed},
        "self_check": check,
        "optimum": best,
        "optimum_grade": best_grade,
        "hs_feasibility": {"E_hs_hi_max_over_grid_mpa": hs_max,
                           "target_below_hs_upper_bound": bool(hs_max >= e_target),
                           "at_optimum": hs_at_best},
        "uncertainty": spread,
        "grade_table": grades,
        "pareto_front": front,
        "n_grid_points": len(rows),
        "n_feasible": sum(1 for r in rows if r["feasible"]),
        "_rows": rows,
    }


def validate(results: dict) -> list[str]:
    """Return the list of failed physics / protocol checks (empty list = all checks pass)."""
    fails: list[str] = []
    sc = results["self_check"]
    tgt = results["inputs"]["E_target_mpa"]

    if abs(sc["E_at_vf0_mpa"] - sc["expected_matrix_E_mpa"]) > 1e-6 * sc["expected_matrix_E_mpa"]:
        fails.append(f"self-check: E(vf=0)={sc['E_at_vf0_mpa']:.3f} != matrix E={sc['expected_matrix_E_mpa']}")
    if abs(sc["rho_at_vf0_g_cc"] - sc["expected_matrix_rho_g_cc"]) > 1e-9:
        fails.append("self-check: rho(vf=0) != matrix density")
    if sc["matrix_alone_meets_target"]:
        fails.append("self-check: neat matrix should be INFEASIBLE against the target but is not")

    opt = results["optimum"]
    if opt is None:
        fails.append("no feasible design found within eta/vf limits")
        return fails

    if opt["E_mt_mpa"] < tgt - 1e-9:
        fails.append(f"optimum modulus {opt['E_mt_mpa']:.1f} MPa < target {tgt} MPa")
    if opt["vf"] > RCP + 1e-12:
        fails.append(f"optimum vf={opt['vf']:.3f} exceeds random close packing {RCP}")
    if opt["vf"] > results["inputs"]["vf_max"] + 1e-12:
        fails.append("optimum vf exceeds requested vf_max")
    if not 0.0 <= opt["eta"] < 1.0:
        fails.append(f"eta={opt['eta']} outside physical range [0,1)")
    if results["inputs"]["grade_fixed"] is None and not (ETA_MIN - 1e-12 <= opt["eta"] <= ETA_MAX + 1e-12):
        fails.append("optimum eta outside the requested [0.80, 0.97] window")
    if not opt["inside_hs_band"]:
        fails.append("HP-MT estimate lies outside the Hashin-Shtrikman band")
    if not results["hs_feasibility"]["target_below_hs_upper_bound"]:
        fails.append("target modulus exceeds the HS upper bound: unreachable by any microstructure")
    if opt["rho_g_cc"] >= results["inputs"]["matrix_rho_g_cc"]:
        fails.append("optimum is not lighter than the neat matrix")
    if opt["rho_g_cc"] <= 0:
        fails.append("non-physical (non-positive) composite density")

    # Every grid estimate must sit inside its own HS band.
    bad = [r for r in results["_rows"] if not r["inside_hs_band"]]
    if bad:
        fails.append(f"{len(bad)} grid estimates fall outside their HS bounds")
    return fails


def write_outputs(results: dict, out_dir: str) -> dict[str, str]:
    """Write results.json, results.csv and the density-vs-modulus trade-off PNG."""
    os.makedirs(out_dir, exist_ok=True)
    rows = results["_rows"]
    public = {k: v for k, v in results.items() if not k.startswith("_")}

    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as fh:
        json.dump(public, fh, indent=2, default=float)

    grid = pd.DataFrame(rows)
    grid.insert(0, "table", "grid")
    front = pd.DataFrame(results["pareto_front"])
    front.insert(0, "table", "pareto_front")
    grades = pd.DataFrame(results["grade_table"])
    grades.insert(0, "table", "grade_table")
    csv_path = os.path.join(out_dir, "results.csv")
    pd.concat([grid, front, grades], ignore_index=True).to_csv(csv_path, index=False)

    png_path = os.path.join(out_dir, "tradeoff.png")
    opt = results["optimum"]
    tgt = results["inputs"]["E_target_mpa"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for eta in sorted({round(r["eta"], 4) for r in rows})[::8]:
        sub = sorted([r for r in rows if abs(r["eta"] - eta) < 1e-9], key=lambda r: r["rho_g_cc"])
        ax[0].plot([r["rho_g_cc"] for r in sub], [r["E_mt_mpa"] for r in sub],
                   lw=0.9, alpha=0.5, label=f"eta={eta:.3f}")
    pf = results["pareto_front"]
    ax[0].plot([r["rho_g_cc"] for r in pf], [r["E_mt_mpa"] for r in pf], "k-", lw=2,
               label="Pareto front")
    ax[0].axhline(tgt, color="r", ls="--", lw=1.2, label=f"target {tgt:.0f} MPa")
    if opt:
        ax[0].plot(opt["rho_g_cc"], opt["E_mt_mpa"], "r*", ms=15, label="optimum")
    ax[0].set_xlabel("density (g/cm$^3$)")
    ax[0].set_ylabel("compressive Young's modulus $E$ (MPa)")
    ax[0].set_title("Trade-off: density vs modulus (HP-MT)")
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(alpha=0.3)

    if opt:
        sub = sorted([r for r in rows if abs(r["eta"] - opt["eta"]) < 1e-9], key=lambda r: r["vf"])
        ax[1].fill_between([r["vf"] for r in sub], [r["E_hs_lo_mpa"] for r in sub],
                           [r["E_hs_hi_mpa"] for r in sub], color="0.85", label="HS bounds")
        ax[1].plot([r["vf"] for r in sub], [r["E_mt_mpa"] for r in sub], "b-o", ms=3, label="HP-MT")
        ax[1].plot([r["vf"] for r in sub], [r["E_ds_mpa"] for r in sub], "g--", label="HP-DS")
        ax[1].axhline(tgt, color="r", ls="--", lw=1.2, label="target")
        ax[1].plot(opt["vf"], opt["E_mt_mpa"], "r*", ms=15)
        ax[1].set_yscale("log")
        ax[1].set_xlabel("particle volume fraction $v_f$")
        ax[1].set_ylabel("$E$ (MPa)")
        ax[1].set_title(f"At optimum eta = {opt['eta']:.4f}: model spread and HS band")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return {"json": json_path, "csv": csv_path, "png": png_path}


def summarise(results: dict, fails: list[str]) -> str:
    """Build the short physics sanity summary printed at the end of the run."""
    sc, opt, unc = results["self_check"], results["optimum"], results["uncertainty"]
    tgt = results["inputs"]["E_target_mpa"]
    L = ["", "=" * 78, "PHYSICS SANITY SUMMARY  (units: MPa, g/cm^3)", "=" * 78,
         f"Model: {results['model']['primary']}",
         f"  cross-check: {results['model']['cross_check']}; envelope: {results['model']['bounds']}",
         "",
         f"Self-check   E(vf=0) = {sc['E_at_vf0_mpa']:.3f} MPa vs neat epoxy {sc['expected_matrix_E_mpa']:.1f} MPa"
         f"  ->  {'OK' if abs(sc['E_at_vf0_mpa'] - sc['expected_matrix_E_mpa']) < 1e-6 else 'FAIL'}",
         f"             matrix alone reaches target? {sc['matrix_alone_meets_target']} "
         f"(3000 < {tgt:.0f} MPa, so reinforcement is required)"]
    if opt is None:
        L += ["", "NO FEASIBLE DESIGN within the requested eta / vf window."]
    else:
        gr = results["optimum_grade"]
        L += ["",
              f"Optimum      eta = {opt['eta']:.4f}  (particle true density {opt['particle_true_density_g_cc']:.3f} g/cm^3)",
              f"             vf  = {opt['vf']:.4f}   (RCP limit {RCP}, requested max {results['inputs']['vf_max']})",
              f"             rho = {opt['rho_g_cc']:.4f} g/cm^3  "
              f"({100 * (1 - opt['rho_g_cc'] / results['inputs']['matrix_rho_g_cc']):.1f} % lighter than neat epoxy)",
              f"             E   = {opt['E_mt_mpa']:.1f} MPa  (target {tgt:.0f} MPa, margin "
              f"{opt['E_mt_mpa'] - tgt:+.1f} MPa)",
              f"             nu  = {opt['nu_mt']:.4f}; specific E = {opt['specific_E_mpa_cc_g']:.0f} MPa cm^3/g",
              "",
              f"HS check     E_lo = {opt['E_hs_lo_mpa']:.1f} <= E_HP-MT = {opt['E_mt_mpa']:.1f} "
              f"<= E_hi = {opt['E_hs_hi_mpa']:.1f} MPa  -> {'inside' if opt['inside_hs_band'] else 'OUTSIDE'} band",
              f"             target {tgt:.0f} MPa is below the HS upper bound "
              f"(max over grid {results['hs_feasibility']['E_hs_hi_max_over_grid_mpa']:.0f} MPa) -> attainable",
              "",
              f"Uncertainty  HP-DS gives {unc['E_ds_mpa']:.1f} MPa at the optimum "
              f"({unc['rel_diff_pct']:.1f} % from HP-MT); DS meets target: {unc['ds_meets_target']}",
              f"             HS band width at optimum = {unc['hs_band_width_pct']:.1f} % of E",
              f"             {unc['note']}"]
        if gr:
            L += ["",
                  f"Catalogue    lightest feasible 3M grade: {gr['grade']} (eta={gr['eta']:.4f}) at "
                  f"vf={gr['vf']:.3f} -> rho={gr['rho_g_cc']:.4f} g/cm^3, E={gr['E_mt_mpa']:.1f} MPa"]
        L += ["",
              f"Trade-off    {results['n_feasible']}/{results['n_grid_points']} grid designs feasible; "
              f"Pareto front has {len(results['pareto_front'])} points (see tradeoff.png)."]
    L += ["", f"Validation   {'ALL CHECKS PASSED' if not fails else 'FAILED: ' + '; '.join(fails)}", "=" * 78]
    return "\n".join(L)


def build_parser() -> argparse.ArgumentParser:
    """CLI whose defaults reproduce the requested task exactly."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matrix", default="epoxy", choices=sorted(MATERIALS),
                    help="matrix material key (default: epoxy, E=3000 MPa, nu=0.35, rho=1.18)")
    ap.add_argument("--grade", default=None,
                    help="fix the microballoon to a 3M grade (K1..S60); default: sweep eta in [0.80,0.97]")
    ap.add_argument("--vf-max", type=float, default=0.60, help="maximum particle volume fraction (default 0.60)")
    ap.add_argument("--e-target", type=float, default=E_TARGET_MPA, help="target modulus in MPa (default 3500)")
    ap.add_argument("--out-dir", default=".", help="output directory (default: current directory)")
    ap.add_argument("--seed", type=int, default=SEED, help="deterministic seed (default 0)")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point: compute, validate, write outputs, print the summary; return the exit code."""
    args = build_parser().parse_args(argv)
    results = compute(matrix_name=args.matrix, grade=args.grade, vf_max=args.vf_max,
                      e_target=args.e_target, seed=args.seed)
    fails = validate(results)
    results["validation"] = {"passed": not fails, "failed_checks": fails}
    paths = write_outputs(results, args.out_dir)
    print(summarise(results, fails))
    print("\nWrote: " + ", ".join(f"{k}={v}" for k, v in paths.items()))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
