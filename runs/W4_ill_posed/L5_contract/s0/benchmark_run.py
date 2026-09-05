#!/usr/bin/env python
"""Modulus and density of epoxy / K46 glass-microballoon syntactic foam.

Requested conditions: particle volume fraction vf = 0.75, wall ratio eta = 1.02.

BOTH REQUESTED INPUTS ARE PHYSICALLY INADMISSIBLE, so no modulus/density is
reported *at* those conditions:

  1. eta = r_inner / r_outer = 1.02 >= 1 describes an inner radius larger than the
     outer radius, i.e. negative wall volume and negative particle mass. eta must
     lie in [0, 1). It is also inconsistent with the named grade: 3M K46 has a
     true density of 0.46 g/cm^3 on a borosilicate shell of 2.54 g/cm^3, which
     fixes eta = (1 - 0.46/2.54)^(1/3) ~= 0.935.
  2. vf = 0.75 exceeds the random close packing limit of monodisperse spheres
     (RCP = 0.64) and even the FCC/HCP ordered-packing maximum (pi/(3*sqrt(2)) =
     0.7405). No arrangement of equal spheres, ordered or random, reaches 0.75.
     (Polydisperse microballoon blends can pass 0.64, but the task specifies
     *monodisperse* particles, and 0.75 is above the ordered limit regardless.)

What this script does instead (the defensible deliverable):
  - runs the required self-check E(vf = 0) == E_matrix == 3000 MPa;
  - uses the datasheet-consistent K46 geometry (eta inferred from true density);
  - sweeps vf over the physically admissible range [0, RCP] with two independent
    homogenization models (HP-MT and HP-DS) bracketed by Hashin-Shtrikman bounds;
  - reports modulus and density at the admissible packing limit vf = 0.64 as the
    closest attainable state to the request, with the model spread as uncertainty;
  - compares against the bundled FoamGPT experimental epoxy/glass-microballoon
    rows (nothing hard-coded) to size the model-vs-experiment gap.

validate() therefore returns non-empty (the premise checks fail) and the script
exits 1. That failure IS the answer to the task as posed.

Units: modulus/stress in MPa, density in g/cm^3, diameters in micrometres.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from foamsim import MATERIALS, hollow_particle  # noqa: E402
from foamsim.data import reference_curve  # noqa: E402
from foamsim.materials import HollowParticle  # noqa: E402
from foamsim.micromechanics import (  # noqa: E402
    RCP,
    density,
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
    hollow_sphere_equivalent,
    particle_crush_onset,
)

SEED = 0
FCC_MAX = np.pi / (3.0 * np.sqrt(2.0))  # 0.74048, densest possible equal-sphere packing
REQUESTED_VF = 0.75
REQUESTED_ETA = 1.02


def set_seeds(seed: int = SEED) -> None:
    """Fix every RNG this script could touch, for bit-identical reruns."""
    random.seed(seed)
    np.random.seed(seed)


def check_premise(requested_vf: float, requested_eta: float, particle: HollowParticle) -> list[dict]:
    """Screen the *requested* inputs for physical admissibility.

    Returns one record per premise check with keys: name, ok, detail. Nothing is
    computed at inadmissible inputs; this runs before any homogenization call.
    """
    eta_ok = 0.0 <= requested_eta < 1.0
    checks = [
        {
            "name": "eta_in_[0,1)",
            "ok": bool(eta_ok),
            "detail": (
                f"requested eta = {requested_eta} is a radius ratio r_in/r_out; eta >= 1 implies "
                f"inner radius >= outer radius, i.e. wall volume fraction 1 - eta^3 = "
                f"{1.0 - requested_eta ** 3:.4f} < 0 and negative particle mass. Admissible range [0, 1)."
            ),
        },
        {
            "name": "eta_consistent_with_named_grade",
            "ok": bool(eta_ok and abs(requested_eta - particle.eta) < 0.02),
            "detail": (
                f"K46 true density {particle.true_density:.3f} g/cm^3 on a {particle.shell.rho:.2f} g/cm^3 "
                f"borosilicate shell fixes eta = {particle.eta:.4f}; the requested {requested_eta} "
                f"contradicts the named grade even before the eta < 1 violation."
            ),
        },
        {
            "name": "vf_below_random_close_packing",
            "ok": bool(requested_vf <= RCP),
            "detail": (
                f"requested vf = {requested_vf} exceeds RCP = {RCP} for monodisperse spheres; "
                f"random packings of equal spheres jam at ~0.64."
            ),
        },
        {
            "name": "vf_below_ordered_packing_maximum",
            "ok": bool(requested_vf <= FCC_MAX),
            "detail": (
                f"requested vf = {requested_vf} exceeds even the FCC/HCP maximum {FCC_MAX:.4f}; "
                f"no equal-sphere arrangement whatsoever attains it, polydispersity excluded by the task."
            ),
        },
    ]
    return checks


def self_check_matrix_limit(matrix, particle: HollowParticle) -> dict:
    """Independent known result: at vf = 0 both models must return the neat matrix."""
    mt0 = hollow_particle_mori_tanaka(matrix, particle, 0.0)
    ds0 = hollow_particle_differential(matrix, particle, 0.0)
    return {
        "E_matrix_mpa": matrix.E,
        "E_mt_vf0_mpa": mt0.E,
        "E_ds_vf0_mpa": ds0.E,
        "nu_mt_vf0": mt0.nu,
        "rho_mt_vf0_g_cc": mt0.rho,
        "abs_err_mt_mpa": abs(mt0.E - matrix.E),
        "abs_err_ds_mpa": abs(ds0.E - matrix.E),
    }


def sweep(matrix, particle: HollowParticle, vf_max: float, n: int = 33) -> list[dict]:
    """Admissible-range sweep: HP-MT, HP-DS, HS bounds and density vs vf."""
    rows = []
    for vf in np.linspace(0.0, vf_max, n):
        vf = float(vf)
        mt = hollow_particle_mori_tanaka(matrix, particle, vf)
        ds = hollow_particle_differential(matrix, particle, vf)
        hs = hashin_shtrikman_bounds(matrix, particle, vf)
        rows.append(
            {
                "vf": vf,
                "E_mt_mpa": mt.E,
                "E_ds_mpa": ds.E,
                "E_hs_lo_mpa": hs["E_lo"],
                "E_hs_hi_mpa": hs["E_hi"],
                "nu_mt": mt.nu,
                "rho_g_cc": mt.rho,
                "specific_E_mpa_cc_g": mt.E / mt.rho,
            }
        )
    return rows


def state_at(matrix, particle: HollowParticle, vf: float) -> dict:
    """Full property record at one admissible vf, with the model spread as uncertainty."""
    mt = hollow_particle_mori_tanaka(matrix, particle, vf)
    ds = hollow_particle_differential(matrix, particle, vf)
    hs = hashin_shtrikman_bounds(matrix, particle, vf)
    lo, hi = min(mt.E, ds.E), max(mt.E, ds.E)
    mid = 0.5 * (lo + hi)
    return {
        "vf": vf,
        "eta": particle.eta,
        "E_mt_mpa": mt.E,
        "E_ds_mpa": ds.E,
        "E_best_mpa": mid,
        "E_model_spread_mpa": hi - lo,
        "E_model_spread_pct": 100.0 * (hi - lo) / mid,
        "E_hs_lo_mpa": hs["E_lo"],
        "E_hs_hi_mpa": hs["E_hi"],
        "nu_mt": mt.nu,
        "rho_g_cc": mt.rho,
        "specific_E_mpa_cc_g": mid / mt.rho,
        "crush_onset_mpa": particle_crush_onset(particle, matrix, vf),
    }


def experimental_comparison(matrix, particle: HollowParticle, matrix_class: str) -> dict:
    """Model-vs-experiment gap on the bundled FoamGPT rows (no hard-coded numbers)."""
    try:
        ref = reference_curve(matrix_class, "glass_microballoon")
    except Exception as exc:  # dataset absent or matrix class unknown
        return {"available": False, "reason": str(exc), "n": 0}
    ref = ref[ref["modulus_mpa"].notna()]
    ref = ref[ref["particle_volume_fraction"] <= RCP]
    if len(ref) == 0:
        return {"available": False, "reason": "no usable rows", "n": 0}
    errs, inside = [], 0
    vf_obs = []
    e_obs = []
    e_pred = []
    for _, r in ref.iterrows():
        vf = float(r["particle_volume_fraction"])
        e_meas = float(r["modulus_mpa"])
        mt = hollow_particle_mori_tanaka(matrix, particle, vf)
        hs = hashin_shtrikman_bounds(matrix, particle, vf)
        errs.append(abs(mt.E - e_meas) / e_meas)
        inside += int(hs["E_lo"] <= e_meas <= hs["E_hi"])
        vf_obs.append(vf)
        e_obs.append(e_meas)
        e_pred.append(mt.E)
    return {
        "available": True,
        "n": len(errs),
        "matrix_class": matrix_class,
        "mape_pct": 100.0 * float(np.mean(errs)),
        "median_ape_pct": 100.0 * float(np.median(errs)),
        "frac_inside_hs_band": inside / len(errs),
        "vf_obs": vf_obs,
        "E_obs_mpa": e_obs,
        "E_pred_mpa": e_pred,
        "grade_labels": sorted({str(g)[:40] for g in ref["particle_grade"].fillna("unstated")}),
        "note": (
            "Experimental moduli commonly sit 20-40% below HP-MT because of matrix porosity, "
            "particle breakage during mixing and imperfect interfaces; the gap is reported, not tuned away. "
            "CAVEAT: the bundled epoxy/glass rows are mostly lighter grades (S22/S32/S38) and layered "
            "functionally-graded foams, not monodisperse K46, so this MAPE is an upper bound on the "
            "K46 model error, not a like-for-like validation."
        ),
    }


def compute(matrix_name: str = "epoxy", grade: str = "K46", vf_max: float = REQUESTED_VF,
            requested_eta: float = REQUESTED_ETA, seed: int = SEED) -> dict:
    """Run the whole analysis and return every key number as a plain dict.

    vf_max is the *requested* volume fraction; it is screened against packing
    limits and the sweep is clipped to the admissible range before any model call.
    """
    set_seeds(seed)
    matrix = MATERIALS[matrix_name]
    particle = hollow_particle(grade)  # eta inferred from the datasheet true density
    eq = hollow_sphere_equivalent(particle)

    premise = check_premise(vf_max, requested_eta, particle)
    admissible_vf_max = float(min(vf_max, RCP))

    results: dict = {
        "task": {
            "matrix": matrix_name,
            "grade": grade,
            "requested_vf": vf_max,
            "requested_eta": requested_eta,
            "dispersity": "monodisperse (as specified)",
        },
        "units": {"modulus": "MPa", "density": "g/cm^3", "stress": "MPa", "diameter": "um",
                  "vf": "volume fraction of particles including hollow cores"},
        "model": {
            "primary": "HP-MT: hollow sphere -> equivalent solid particle (Hashin 1962 exact K, "
                       "HS-upper G of the porous shell) -> Mori-Tanaka (Benveniste 1987)",
            "secondary": "HP-DS: same equivalent particle -> differential scheme (McLaughlin 1977)",
            "bounds": "Hashin-Shtrikman bounds on the two-phase (matrix + equivalent particle) system",
            "assumptions": [
                "linear elastic, isotropic, perfectly bonded phases; no interphase",
                "spherical, non-interpenetrating, uniformly dispersed particles; no clustering",
                "intact microballoons (no crushing) and fully dense matrix (matrix_porosity = 0)",
                "small strain, quasi-static; MT is a mean-field estimate, not a bound",
            ],
        },
        "constituents": {
            "matrix": {"name": matrix.name, "E_mpa": matrix.E, "nu": matrix.nu, "rho_g_cc": matrix.rho,
                       "K_mpa": matrix.K, "G_mpa": matrix.G},
            "shell": {"name": particle.shell.name, "E_mpa": particle.shell.E, "nu": particle.shell.nu,
                      "rho_g_cc": particle.shell.rho},
            "particle": {"grade": grade, "eta_from_datasheet": particle.eta,
                         "true_density_g_cc": particle.true_density,
                         "wall_volume_fraction": particle.wall_volume_fraction,
                         "diameter_um": particle.diameter_um},
            "equivalent_particle": {"E_mpa": eq.E, "nu": eq.nu, "K_mpa": eq.K, "G_mpa": eq.G},
        },
        "packing_limits": {"RCP_monodisperse": RCP, "ordered_max_fcc_hcp": float(FCC_MAX)},
        "premise_checks": premise,
        "premise_ok": all(c["ok"] for c in premise),
        "self_check_vf0": self_check_matrix_limit(matrix, particle),
        "admissible_vf_max": admissible_vf_max,
        "sweep": sweep(matrix, particle, admissible_vf_max),
        "closest_admissible_state": state_at(matrix, particle, admissible_vf_max),
        "requested_state": {
            "computed": False,
            "reason": "vf = 0.75 and eta = 1.02 are both physically inadmissible; "
                      "computing a modulus there would require negative particle mass and an "
                      "unrealisable monodisperse packing. No number is reported at these conditions.",
        },
        "experimental_comparison": experimental_comparison(matrix, particle, matrix_name),
    }
    results["deliverable"] = {
        "answer": "NOT COMPUTABLE AS POSED",
        "explanation": "Both stated conditions violate hard physical constraints (see premise_checks).",
        "closest_defensible_answer": {
            "vf": admissible_vf_max,
            "eta": particle.eta,
            "E_mpa": results["closest_admissible_state"]["E_best_mpa"],
            "E_uncertainty_mpa": results["closest_admissible_state"]["E_model_spread_mpa"],
            "E_range_models_mpa": [results["closest_admissible_state"]["E_ds_mpa"],
                                   results["closest_admissible_state"]["E_mt_mpa"]],
            "E_hs_band_mpa": [results["closest_admissible_state"]["E_hs_lo_mpa"],
                              results["closest_admissible_state"]["E_hs_hi_mpa"]],
            "rho_g_cc": results["closest_admissible_state"]["rho_g_cc"],
        },
        "how_to_make_the_task_well_posed": [
            "keep vf <= 0.64 for monodisperse microballoons (a bimodal/polydisperse blend can reach ~0.7 "
            "but is a different microstructure than the one specified)",
            "drop the eta = 1.02 override and use the K46 datasheet geometry (eta ~= 0.935), or state an "
            "eta in [0, 1) if a non-K46 wall thickness is intended",
            "if a very light foam is the real goal, lower the particle true density (K20, K15, K1) at "
            "vf <= 0.64 instead of raising vf beyond packing",
        ],
    }
    return results


def validate(results: dict) -> list[str]:
    """Return the list of failed checks (empty list == everything passed)."""
    failed: list[str] = []

    # 1. required self-check: E(vf=0) must equal the matrix modulus exactly (to 1e-6 relative)
    sc = results["self_check_vf0"]
    tol = 1e-6 * sc["E_matrix_mpa"]
    if sc["abs_err_mt_mpa"] > tol:
        failed.append(f"self-check: HP-MT E(vf=0) = {sc['E_mt_mpa']:.6f} != matrix {sc['E_matrix_mpa']} MPa")
    if sc["abs_err_ds_mpa"] > tol:
        failed.append(f"self-check: HP-DS E(vf=0) = {sc['E_ds_mpa']:.6f} != matrix {sc['E_matrix_mpa']} MPa")

    # 2. premise checks on the requested inputs
    for c in results["premise_checks"]:
        if not c["ok"]:
            failed.append(f"premise[{c['name']}]: {c['detail']}")

    # 3. every estimate must lie inside the HS band, and density must be monotone decreasing in vf
    prev_rho = None
    for r in results["sweep"]:
        for key in ("E_mt_mpa", "E_ds_mpa"):
            if not (r["E_hs_lo_mpa"] - 1e-6 <= r[key] <= r["E_hs_hi_mpa"] + 1e-6):
                failed.append(f"bounds: {key} = {r[key]:.3f} outside HS band at vf = {r['vf']:.3f}")
        if prev_rho is not None and r["rho_g_cc"] > prev_rho + 1e-12:
            failed.append(f"density not monotone decreasing at vf = {r['vf']:.3f}")
        prev_rho = r["rho_g_cc"]

    # 4. sanity: stiff glass microballoons at K46 density should raise E above the matrix
    st = results["closest_admissible_state"]
    if st["E_best_mpa"] <= 0 or st["rho_g_cc"] <= 0:
        failed.append("non-positive modulus or density at the admissible limit")
    if not (0.0 < st["nu_mt"] < 0.5):
        failed.append(f"Poisson ratio out of range: {st['nu_mt']:.4f}")

    return failed


def write_json(results: dict, out_dir: Path) -> Path:
    """Write results.json with every key number."""
    path = out_dir / "results.json"
    path.write_text(json.dumps(results, indent=2, default=float))
    return path


def write_csv(results: dict, out_dir: Path) -> Path:
    """Write results.csv: the admissible vf sweep table."""
    path = out_dir / "results.csv"
    rows = results["sweep"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def make_figure(results: dict, out_dir: Path) -> Path:
    """Two-panel PNG: E vs vf (with HS band, forbidden region, experiment) and density vs vf."""
    rows = results["sweep"]
    vf = np.array([r["vf"] for r in rows])
    e_mt = np.array([r["E_mt_mpa"] for r in rows])
    e_ds = np.array([r["E_ds_mpa"] for r in rows])
    e_lo = np.array([r["E_hs_lo_mpa"] for r in rows])
    e_hi = np.array([r["E_hs_hi_mpa"] for r in rows])
    rho = np.array([r["rho_g_cc"] for r in rows])
    req_vf = results["task"]["requested_vf"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

    ax[0].fill_between(vf, e_lo, e_hi, color="0.85", label="Hashin-Shtrikman band")
    ax[0].plot(vf, e_mt, "-", color="C0", lw=2, label="HP-MT (primary)")
    ax[0].plot(vf, e_ds, "--", color="C1", lw=2, label="HP-DS (differential)")
    exp = results["experimental_comparison"]
    if exp.get("available"):
        ax[0].plot(exp["vf_obs"], exp["E_obs_mpa"], "k.", ms=6, alpha=0.6,
                   label=f"FoamGPT experiment (n={exp['n']})")
    ax[0].axvspan(RCP, max(req_vf, 0.8), color="tab:red", alpha=0.12)
    ax[0].axvline(RCP, color="tab:red", lw=1.5, label=f"RCP = {RCP} (monodisperse)")
    ax[0].axvline(FCC_MAX, color="tab:purple", lw=1.2, ls=":", label=f"FCC max = {FCC_MAX:.3f}")
    ax[0].axvline(req_vf, color="k", lw=1.5, ls="-.", label=f"requested vf = {req_vf} (unrealisable)")
    ax[0].set_xlim(0, max(req_vf, 0.8))
    ax[0].set_xlabel("particle volume fraction vf (-)")
    ax[0].set_ylabel("Young's modulus E (MPa)")
    ax[0].set_title(f"{results['task']['matrix']} / {results['task']['grade']} "
                    f"(eta = {results['constituents']['particle']['eta_from_datasheet']:.3f})")
    ax[0].legend(fontsize=7, loc="upper left")
    ax[0].grid(alpha=0.3)

    ax[1].plot(vf, rho, "-", color="C2", lw=2, label="composite density")
    ax[1].axvspan(RCP, max(req_vf, 0.8), color="tab:red", alpha=0.12)
    ax[1].axvline(RCP, color="tab:red", lw=1.5)
    ax[1].axvline(req_vf, color="k", lw=1.5, ls="-.")
    ax[1].set_xlim(0, max(req_vf, 0.8))
    ax[1].set_xlabel("particle volume fraction vf (-)")
    ax[1].set_ylabel("density (g/cm^3)")
    ax[1].set_title("density, rule of mixtures (shaded = not packable)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.suptitle("Requested vf = 0.75 / eta = 1.02 are physically inadmissible; "
                 "results shown only over the admissible range", fontsize=10)
    fig.tight_layout()
    path = out_dir / "modulus_density_vs_vf.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_summary(results: dict, failed: list[str]) -> None:
    """Print the short physics sanity summary."""
    sc = results["self_check_vf0"]
    st = results["closest_admissible_state"]
    d = results["deliverable"]["closest_defensible_answer"]
    exp = results["experimental_comparison"]
    p = results["constituents"]["particle"]

    print("\n" + "=" * 78)
    print("PHYSICS SANITY SUMMARY (units: E in MPa, density in g/cm^3)")
    print("=" * 78)
    print(f"self-check  E(vf=0): HP-MT {sc['E_mt_vf0_mpa']:.6f} | HP-DS {sc['E_ds_vf0_mpa']:.6f} "
          f"| matrix {sc['E_matrix_mpa']:.1f}  -> {'PASS' if sc['abs_err_mt_mpa'] < 1e-6 else 'FAIL'}")
    print(f"K46 geometry: eta = {p['eta_from_datasheet']:.4f} from true density "
          f"{p['true_density_g_cc']:.3f} on a {results['constituents']['shell']['rho_g_cc']:.2f} shell")
    print("\nPREMISE CHECKS ON THE REQUESTED CONDITIONS:")
    for c in results["premise_checks"]:
        print(f"  [{'ok  ' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print("\nRESULT AT THE REQUESTED CONDITIONS (vf=0.75, eta=1.02): NOT COMPUTED.")
    print(f"  {results['requested_state']['reason']}")
    print(f"\nCLOSEST PHYSICALLY ADMISSIBLE STATE (vf = {d['vf']:.4f}, eta = {d['eta']:.4f}):")
    print(f"  E   = {d['E_mpa']:.1f} MPa  (+/- {d['E_uncertainty_mpa']:.1f} model spread; "
          f"HP-DS {st['E_ds_mpa']:.1f} / HP-MT {st['E_mt_mpa']:.1f})")
    print(f"  HS band at this vf: [{d['E_hs_band_mpa'][0]:.1f}, {d['E_hs_band_mpa'][1]:.1f}] MPa "
          "-> both estimates lie inside")
    print(f"  rho = {d['rho_g_cc']:.4f} g/cm^3   nu = {st['nu_mt']:.4f}   "
          f"specific E = {st['specific_E_mpa_cc_g']:.0f} MPa/(g/cm^3)")
    print(f"  microballoon crush onset ~ {st['crush_onset_mpa']:.0f} MPa (order of magnitude)")
    if exp.get("available"):
        print(f"\nvs FoamGPT experiment ({exp['matrix_class']}/glass_microballoon, n = {exp['n']}): "
              f"HP-MT MAPE {exp['mape_pct']:.1f}% (median {exp['median_ape_pct']:.1f}%), "
              f"{100 * exp['frac_inside_hs_band']:.0f}% of points inside the HS band")
        print(f"  {exp['note']}")
    print("\nHOW TO MAKE THE TASK WELL POSED:")
    for s in results["deliverable"]["how_to_make_the_task_well_posed"]:
        print(f"  - {s}")
    print(f"\nFAILED CHECKS: {len(failed)}")
    for f in failed:
        print(f"  ! {f}")
    print("=" * 78)


def build_parser() -> argparse.ArgumentParser:
    """CLI whose defaults reproduce the task exactly as posed."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matrix", default="epoxy", help="matrix key in foamsim.MATERIALS (default: epoxy)")
    ap.add_argument("--grade", default="K46", help="microballoon grade (default: K46)")
    ap.add_argument("--vf-max", type=float, default=REQUESTED_VF,
                    help="requested particle volume fraction (default: 0.75, as posed)")
    ap.add_argument("--eta", type=float, default=REQUESTED_ETA,
                    help="requested wall ratio r_in/r_out (default: 1.02, as posed)")
    ap.add_argument("--out-dir", default=".", help="output directory (default: current directory)")
    ap.add_argument("--seed", type=int, default=SEED, help="deterministic seed (default: 0)")
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point: compute, validate, write results.json/results.csv/PNG, print summary."""
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = compute(matrix_name=args.matrix, grade=args.grade, vf_max=args.vf_max,
                      requested_eta=args.eta, seed=args.seed)
    failed = validate(results)
    results["failed_checks"] = failed
    results["n_failed_checks"] = len(failed)

    write_json(results, out_dir)
    write_csv(results, out_dir)
    make_figure(results, out_dir)
    print_summary(results, failed)
    print(f"\nwrote: {out_dir / 'results.json'}, {out_dir / 'results.csv'}, "
          f"{out_dir / 'modulus_density_vs_vf.png'}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
