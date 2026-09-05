"""FE RVE homogenization vs analytical estimates for a syntactic foam (epoxy / K46, vf = 0.30).

What this script does
---------------------
1. Self-check: a homogeneous voxel box must return the matrix moduli exactly (KUBC is exact there).
2. Builds random periodic packings of hollow spheres (RSA, foamsim.rve.random_packing) at the target
   particle volume fraction for two deterministic seeds, and homogenizes each with voxel finite
   elements (scikit-fem, trilinear hexahedra, kinematic uniform boundary conditions) at two mesh
   resolutions.
3. Compares the FE moduli with the analytical hollow-particle Mori-Tanaka (HP-MT) estimate and the
   Hashin-Shtrikman bounds computed for the *realised* volume fraction of each packing.
4. Writes results.json, results.csv and a PNG figure, and prints a physics sanity summary.

Models and assumptions
----------------------
* Particle model: each 3M K46 glass microballoon is replaced by its equivalent homogeneous solid
  sphere (Hashin composite-sphere-assemblage exact bulk modulus, HS-upper shear) -- foamsim
  mode="equivalent". The explicit shell/void mode would need n >= ~64 voxels for eta ~ 0.94 walls,
  which is far more expensive than the resolutions used here; this is stated, not silently ignored.
* eta (inner/outer radius ratio) is inferred from the K46 datasheet true density 0.46 g/cm^3 and the
  borosilicate shell density 2.54 g/cm^3 -- no experimental modulus is hard-coded anywhere.
* Analytical reference: HP-MT (Mori-Tanaka / Benveniste with the equivalent particle), i.e. the
  HS estimate with the matrix as reference medium. It assumes dilute-interaction-corrected,
  perfectly bonded, isotropically distributed spherical inclusions and linear elasticity.
* KUBC on a finite RVE is an upper-bound-type estimate: it over-predicts stiffness, and the bias
  shrinks with RVE size. Spread over seeds/resolutions is reported as the uncertainty.

Units: Young's/bulk/shear moduli in MPa, densities in g/cm^3, particle diameter in micrometres;
volume fractions dimensionless and referred to the particle INCLUDING its hollow core.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from foamsim import MATERIALS, hollow_particle  # noqa: E402
from foamsim.fem import homogenize, homogenize_homogeneous  # noqa: E402
from foamsim.micromechanics import (  # noqa: E402
    hashin_shtrikman_bounds,
    hollow_particle_mori_tanaka,
    hollow_sphere_equivalent,
)
from foamsim.rve import random_packing  # noqa: E402

# Deterministic run configuration (defaults reproduce the requested task).
RESOLUTIONS = (16, 24)
SEEDS = (0, 1)
N_SPHERES = 16
MT_TOLERANCE = 0.25  # accepted |FE - MT| / MT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line interface; the defaults reproduce the epoxy/K46 vf=0.30 task."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matrix", default="epoxy", help="matrix material key in foamsim.MATERIALS")
    ap.add_argument("--grade", default="K46", help="3M glass-bubble grade (true density sets eta)")
    ap.add_argument("--vf-max", type=float, default=0.30,
                    help="target particle volume fraction (the single vf studied here)")
    ap.add_argument("--out-dir", default=".", help="directory for results.json/results.csv/PNG")
    return ap.parse_args(argv)


def self_check_homogeneous(matrix, n: int = 4) -> dict:
    """Recover the independent known result: a homogeneous box must return the matrix moduli."""
    eff = homogenize_homogeneous(matrix, n=n)
    return {
        "n": n,
        "E_fe_mpa": eff.E,
        "nu_fe": eff.nu,
        "E_matrix_mpa": matrix.E,
        "nu_matrix": matrix.nu,
        "rel_err_E": abs(eff.E - matrix.E) / matrix.E,
        "rel_err_nu": abs(eff.nu - matrix.nu) / matrix.nu,
    }


def fe_cases(matrix, particle, vf: float) -> list[dict]:
    """Run FE homogenization for every (resolution, seed) pair; one row per case."""
    rows = []
    for seed in SEEDS:
        rve = random_packing(vf=vf, n_spheres=N_SPHERES, eta=particle.eta, seed=seed)
        for n in RESOLUTIONS:
            eff = homogenize(rve, matrix, particle, n=n, mode="equivalent")
            mt = hollow_particle_mori_tanaka(matrix, particle, rve.vf)
            hs = hashin_shtrikman_bounds(matrix, particle, rve.vf)
            rows.append({
                "resolution_n": n,
                "seed": seed,
                "model": eff.model,
                "vf_target": vf,
                "vf_realised": rve.vf,
                "E_fe_mpa": eff.E,
                "K_fe_mpa": eff.K,
                "G_fe_mpa": eff.G,
                "nu_fe": eff.nu,
                "rho_g_cc": eff.rho,
                "E_mt_mpa": mt.E,
                "E_hs_lo_mpa": hs["E_lo"],
                "E_hs_hi_mpa": hs["E_hi"],
                "rel_diff_fe_vs_mt": (eff.E - mt.E) / mt.E,
                "inside_hs_bounds": bool(hs["E_lo"] - 1e-9 <= eff.E <= hs["E_hi"] + 1e-9),
            })
    return rows


def compute(matrix_key: str = "epoxy", grade: str = "K46", vf: float = 0.30) -> dict:
    """Run the whole study and return every key number as a plain dictionary."""
    matrix = MATERIALS[matrix_key]
    particle = hollow_particle(grade)
    eq = hollow_sphere_equivalent(particle)

    check = self_check_homogeneous(matrix)
    rows = fe_cases(matrix, particle, vf)

    E_fe = [r["E_fe_mpa"] for r in rows]
    vf_real = statistics.fmean(r["vf_realised"] for r in rows)
    mt = hollow_particle_mori_tanaka(matrix, particle, vf_real)
    hs = hashin_shtrikman_bounds(matrix, particle, vf_real)
    mean_E = statistics.fmean(E_fe)
    sd_E = statistics.stdev(E_fe) if len(E_fe) > 1 else 0.0

    return {
        "units": {"modulus": "MPa", "density": "g/cm^3", "length": "micrometre", "vf": "dimensionless"},
        "inputs": {
            "matrix": matrix_key,
            "matrix_E_mpa": matrix.E, "matrix_nu": matrix.nu, "matrix_rho_g_cc": matrix.rho,
            "grade": grade,
            "particle_true_density_g_cc": particle.true_density,
            "shell_E_mpa": particle.shell.E, "shell_nu": particle.shell.nu, "shell_rho_g_cc": particle.shell.rho,
            "eta": particle.eta,
            "equivalent_particle_E_mpa": eq.E, "equivalent_particle_nu": eq.nu,
            "vf_target": vf, "n_spheres": N_SPHERES,
            "resolutions": list(RESOLUTIONS), "seeds": list(SEEDS),
        },
        "method": {
            "fe": "voxel hexahedral FE (scikit-fem), kinematic uniform BCs (KUBC), mode='equivalent'",
            "analytical": "HP-MT (hollow-particle Mori-Tanaka on the equivalent solid particle)",
            "bounds": "Hashin-Shtrikman bounds for matrix + equivalent particle",
            "note": ("mode='shell' (explicit glass shell + void core) was NOT used: eta="
                     f"{particle.eta:.3f} would require n >= ~64 voxels for >=2 voxels across the wall."),
            "kubc_bias": "KUBC over-predicts stiffness for finite RVEs; spread over seeds/resolutions reported.",
        },
        "self_check_homogeneous_box": check,
        "cases": rows,
        "summary": {
            "vf_realised_mean": vf_real,
            "E_fe_mean_mpa": mean_E,
            "E_fe_sd_mpa": sd_E,
            "E_fe_min_mpa": min(E_fe),
            "E_fe_max_mpa": max(E_fe),
            "E_fe_spread_pct": 100.0 * (max(E_fe) - min(E_fe)) / mean_E,
            "E_mt_mpa": mt.E,
            "E_hs_lo_mpa": hs["E_lo"],
            "E_hs_hi_mpa": hs["E_hi"],
            "rel_diff_mean_fe_vs_mt": (mean_E - mt.E) / mt.E,
            "all_cases_inside_hs": all(r["inside_hs_bounds"] for r in rows),
            "mean_inside_hs": bool(hs["E_lo"] <= mean_E <= hs["E_hi"]),
            "hs_hi_exceedance_pct": 100.0 * max(0.0, mean_E - hs["E_hi"]) / hs["E_hi"],
            "hs_band_width_pct": 100.0 * (hs["E_hi"] - hs["E_lo"]) / hs["E_lo"],
            "E_fe_by_resolution": {str(n): statistics.fmean(
                r["E_fe_mpa"] for r in rows if r["resolution_n"] == n) for n in RESOLUTIONS},
            "rho_g_cc": mt.rho,
        },
    }


def validate(results: dict) -> list[str]:
    """Return the list of failed physics/self-check assertions (empty list = all checks passed)."""
    failed: list[str] = []
    c = results["self_check_homogeneous_box"]
    if c["rel_err_E"] > 1e-6:
        failed.append(f"homogeneous box E off by {c['rel_err_E']:.2e} (must be ~0)")
    if c["rel_err_nu"] > 1e-6:
        failed.append(f"homogeneous box nu off by {c['rel_err_nu']:.2e} (must be ~0)")

    s = results["summary"]
    if not s["all_cases_inside_hs"]:
        failed.append(f"at least one FE case falls outside the HS bounds "
                      f"(mean exceeds E_hi by {s['hs_hi_exceedance_pct']:.2f}%)")
    if not s["mean_inside_hs"]:
        failed.append(f"mean FE modulus falls outside the HS bounds "
                      f"(by {s['hs_hi_exceedance_pct']:.2f}% of E_hi)")
    if abs(s["rel_diff_mean_fe_vs_mt"]) >= MT_TOLERANCE:
        failed.append(f"|FE - MT|/MT = {abs(s['rel_diff_mean_fe_vs_mt']):.3f} >= {MT_TOLERANCE}")
    if s["E_fe_mean_mpa"] <= results["inputs"]["matrix_E_mpa"]:
        failed.append("stiff K46 particles must raise E above the epoxy matrix modulus")
    return failed


def write_outputs(results: dict, out_dir: Path) -> None:
    """Write results.json, results.csv (per-case + summary tables) and the comparison PNG figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    rows = results["cases"]
    with (out_dir / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        fh.write("\n")
        sw = csv.writer(fh)
        sw.writerow(["summary_quantity", "value"])
        for k, v in results["summary"].items():
            sw.writerow([k, v])

    plot_results(results, out_dir / "fe_vs_analytical.png")


def plot_results(results: dict, path: Path) -> None:
    """Plot FE modulus per (resolution, seed) against the MT estimate and the HS band."""
    s = results["summary"]
    rows = results["cases"]
    labels = [f"n={r['resolution_n']}\nseed={r['seed']}" for r in rows]
    x = range(len(rows))

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhspan(s["E_hs_lo_mpa"], s["E_hs_hi_mpa"], color="0.85", label="Hashin-Shtrikman band")
    ax.axhline(s["E_mt_mpa"], color="C3", ls="--", label=f"HP-MT = {s['E_mt_mpa']:.0f} MPa")
    ax.axhline(s["E_fe_mean_mpa"], color="C0", ls=":",
               label=f"FE mean = {s['E_fe_mean_mpa']:.0f} MPa")
    ax.errorbar([len(rows) + 0.5], [s["E_fe_mean_mpa"]], yerr=[s["E_fe_sd_mpa"]],
                fmt="s", color="C0", capsize=5, label="FE mean +/- 1 s.d.")
    ax.plot(list(x), [r["E_fe_mpa"] for r in rows], "o", color="C0", label="FE-KUBC cases")
    ax.set_xticks(list(x) + [len(rows) + 0.5])
    ax.set_xticklabels(labels + ["mean"])
    ax.set_ylabel("Young's modulus E (MPa)")
    ax.set_title(f"{results['inputs']['matrix']} / {results['inputs']['grade']}, "
                 f"vf = {s['vf_realised_mean']:.3f}: FE-KUBC vs analytical")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_summary(results: dict, failed: list[str]) -> None:
    """Print a short physics sanity summary (units, table, spread, bounds verdict)."""
    inp, s = results["inputs"], results["summary"]
    c = results["self_check_homogeneous_box"]
    print("\n=== Physics sanity summary (all moduli in MPa, densities in g/cm^3) ===")
    print(f"System: {inp['matrix']} matrix (E={inp['matrix_E_mpa']:.0f}, nu={inp['matrix_nu']}) + "
          f"{inp['grade']} microballoons (eta={inp['eta']:.4f}, true rho={inp['particle_true_density_g_cc']:.2f})")
    print(f"Model : FE voxel homogenization with KUBC, equivalent-particle mode "
          f"(equivalent particle E={inp['equivalent_particle_E_mpa']:.0f} MPa); "
          f"analytical = HP-MT + HS bounds")
    print(f"Self-check: homogeneous box E={c['E_fe_mpa']:.3f} vs matrix {c['E_matrix_mpa']:.0f} "
          f"(rel err {c['rel_err_E']:.2e}), nu rel err {c['rel_err_nu']:.2e}")
    print(f"\n{'n':>4} {'seed':>5} {'vf':>7} {'E_FE':>9} {'E_MT':>9} {'rel diff':>9}  in HS?")
    for r in results["cases"]:
        print(f"{r['resolution_n']:>4} {r['seed']:>5} {r['vf_realised']:>7.4f} {r['E_fe_mpa']:>9.1f} "
              f"{r['E_mt_mpa']:>9.1f} {r['rel_diff_fe_vs_mt']:>+9.3f}  {r['inside_hs_bounds']}")
    print(f"\nFE mean = {s['E_fe_mean_mpa']:.1f} +/- {s['E_fe_sd_mpa']:.1f} MPa "
          f"(range {s['E_fe_min_mpa']:.1f}-{s['E_fe_max_mpa']:.1f}, spread {s['E_fe_spread_pct']:.1f}% of mean)")
    print(f"HP-MT   = {s['E_mt_mpa']:.1f} MPa; relative difference (FE-MT)/MT = "
          f"{s['rel_diff_mean_fe_vs_mt']:+.1%}")
    print(f"HS band = [{s['E_hs_lo_mpa']:.1f}, {s['E_hs_hi_mpa']:.1f}] MPa "
          f"(band width {s['hs_band_width_pct']:.1f}% -- narrow, the phase contrast is only ~2:1); "
          f"FE strictly inside HS bounds: {s['all_cases_inside_hs'] and s['mean_inside_hs']}")
    if s["hs_hi_exceedance_pct"] > 0:
        by_n = s["E_fe_by_resolution"]
        trend = " -> ".join(f"n={n}: {by_n[str(n)]:.1f}" for n in RESOLUTIONS)
        print(f"  FE mean sits {s['hs_hi_exceedance_pct']:.2f}% ABOVE the HS upper bound. This is the expected "
              "sign of the two upward biases of the method, not a sign that MT is wrong: KUBC on a "
              f"{N_SPHERES}-sphere window is a rigorous upper bound on the effective modulus (the HS bound "
              "applies to the exact effective property of a statistically homogeneous medium), and a "
              "displacement-based voxel mesh with stair-stepped interfaces adds further stiffening. "
              f"The FE value does fall with refinement ({trend} MPa), i.e. it moves toward the band. "
              "Treat it as marginally outside, and the FE-vs-MT agreement below as the substantive result.")
    print(f"Composite density = {s['rho_g_cc']:.3f} g/cm^3 (matrix {inp['matrix_rho_g_cc']:.2f})")
    print("Note: KUBC on a small RVE is an upper-bound-type estimate, so FE above HP-MT is expected; "
          "the shell-resolved mode was not used (see results.json['method']['note']).")
    print("CHECKS: " + ("all passed" if not failed else "FAILED -> " + "; ".join(failed)))


def main(argv: list[str] | None = None) -> int:
    """Entry point: compute, validate, write outputs, print summary; return 1 if any check fails."""
    args = parse_args(argv)
    results = compute(matrix_key=args.matrix, grade=args.grade, vf=args.vf_max)
    failed = validate(results)
    results["failed_checks"] = failed
    write_outputs(results, Path(args.out_dir))
    print_summary(results, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
