"""W3: FE RVE homogenization vs analytical estimates, epoxy / 3M K46 at vf = 0.30.

Units throughout: moduli in MPa, densities in g/cm^3, lengths (particle diameter) in micrometres.
vf is the volume fraction of the particles INCLUDING their hollow cores.

Model / assumptions
-------------------
* Numerical: voxel finite-element homogenization (foamsim.fem.homogenize) of a periodic random
  packing of equal hollow spheres, with kinematic uniform boundary conditions (KUBC), six load
  cases -> Voigt stiffness -> closest isotropic (K, G). mode="equivalent": each microballoon is
  replaced by its homogeneous equivalent particle (Hashin's exact hollow-sphere K, plus the
  matching shell G), so the shell/core geometry is not resolved voxel-by-voxel. KUBC on a finite
  RVE is an upper-bound-type estimate: it overestimates stiffness, the more so the coarser the
  mesh and the fewer the spheres.
* Analytical: hollow-particle Mori-Tanaka (HP-MT) on the same equivalent particle, plus the
  Hashin-Shtrikman bounds for the two-phase (matrix + equivalent particle) composite. Both are
  linear-elastic, isotropic, perfectly bonded, no matrix porosity, no particle breakage.
* All constituent numbers come from the foamsim toolkit (MATERIALS["epoxy"], MATERIALS["glass"],
  3M K46 true density -> eta). Nothing is hard-coded from experiment.
"""
from __future__ import annotations

import json
import platform
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from foamsim import MATERIALS, hollow_particle
from foamsim.fem import homogenize, homogenize_homogeneous
from foamsim.micromechanics import (
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
    hollow_sphere_equivalent,
)
from foamsim.rve import random_packing

VF_TARGET = 0.30
N_SPHERES = 16
RESOLUTIONS = (16, 24)
SEEDS = (0, 1)
MT_TOL = 0.25  # self-check: |FE - MT| / MT must be < 25 %


def self_check_homogeneous(matrix, n: int = 4) -> dict:
    """Independent known result: a homogeneous box must return the matrix moduli exactly."""
    eff = homogenize_homogeneous(matrix, n=n)
    rel_E = abs(eff.E - matrix.E) / matrix.E
    rel_nu = abs(eff.nu - matrix.nu) / matrix.nu
    return {
        "n": n,
        "E_fe_mpa": eff.E,
        "E_matrix_mpa": matrix.E,
        "nu_fe": eff.nu,
        "nu_matrix": matrix.nu,
        "rel_diff_E": rel_E,
        "rel_diff_nu": rel_nu,
        "passed": bool(rel_E < 1e-6 and rel_nu < 1e-6),
    }


def main() -> None:
    matrix = MATERIALS["epoxy"]
    part = hollow_particle("K46")
    eq = hollow_sphere_equivalent(part)

    print("=" * 78)
    print("W3  FE RVE homogenization vs analytical  -  epoxy / 3M K46, target vf = %.2f" % VF_TARGET)
    print("=" * 78)
    print(f"matrix   : {matrix.name}  E={matrix.E:.0f} MPa  nu={matrix.nu:.2f}  rho={matrix.rho:.2f} g/cm3")
    print(f"shell    : {part.shell.name}  E={part.shell.E:.0f} MPa  nu={part.shell.nu:.2f}  "
          f"rho={part.shell.rho:.2f} g/cm3")
    print(f"particle : K46  eta={part.eta:.4f}  true density={part.true_density:.3f} g/cm3  "
          f"d={part.diameter_um:.0f} um")
    print(f"equivalent solid particle: E={eq.E:.1f} MPa  nu={eq.nu:.3f}  (K={eq.K:.1f}, G={eq.G:.1f} MPa)")

    # ---------------------------------------------------------------- self-check 1
    print("\n[self-check 1] homogeneous box (no particles) must return the matrix moduli")
    hom = self_check_homogeneous(matrix)
    print(f"  FE n={hom['n']}: E = {hom['E_fe_mpa']:.6f} MPa vs matrix {hom['E_matrix_mpa']:.6f} MPa "
          f"(rel diff {hom['rel_diff_E']:.2e}); nu = {hom['nu_fe']:.6f} vs {hom['nu_matrix']:.6f}  "
          f"-> {'PASS' if hom['passed'] else 'FAIL'}")

    # ---------------------------------------------------------------- FE runs
    runs = []
    realised_vfs = []
    for seed in SEEDS:
        rve = random_packing(vf=VF_TARGET, n_spheres=N_SPHERES, eta=part.eta, seed=seed)
        realised_vfs.append(rve.vf)
        for n in RESOLUTIONS:
            print(f"\n[FE] n={n:>2d} seed={seed}  ({N_SPHERES} spheres, realised vf={rve.vf:.4f}, "
                  f"radius={rve.radius:.4f} box units) ...", flush=True)
            eff = homogenize(rve, matrix, part, n=n, mode="equivalent")
            print(f"     -> model={eff.model}  E={eff.E:.1f} MPa  nu={eff.nu:.4f}  "
                  f"K={eff.K:.1f}  G={eff.G:.1f} MPa  rho={eff.rho:.4f} g/cm3")
            runs.append({"resolution_n": n, "seed": seed, "realised_vf": rve.vf, "model": eff.model,
                         "E_mpa": eff.E, "K_mpa": eff.K, "G_mpa": eff.G, "nu": eff.nu,
                         "rho_g_cc": eff.rho})

    vf_used = float(mean(realised_vfs))
    E_fe = [r["E_mpa"] for r in runs]
    E_mean, E_sd = float(mean(E_fe)), float(pstdev(E_fe))
    E_min, E_max = float(min(E_fe)), float(max(E_fe))

    # ---------------------------------------------------------------- analytical
    mt = hollow_particle_mori_tanaka(matrix, part, vf_used)
    ds = hollow_particle_differential(matrix, part, vf_used)
    hs = hashin_shtrikman_bounds(matrix, part, vf_used)

    for r in runs:
        r["E_mt_mpa"] = mt.E
        r["rel_diff_vs_mt"] = (r["E_mpa"] - mt.E) / mt.E
        r["inside_hs"] = bool(hs["E_lo"] - 1e-9 <= r["E_mpa"] <= hs["E_hi"] + 1e-9)

    rel_mean_vs_mt = (E_mean - mt.E) / mt.E
    all_inside = all(r["inside_hs"] for r in runs)
    mean_inside = bool(hs["E_lo"] <= E_mean <= hs["E_hi"])
    mt_check = bool(abs(rel_mean_vs_mt) < MT_TOL)

    # ---------------------------------------------------------------- outputs
    results = {
        "task": "W3 FE RVE homogenization vs analytical estimate, epoxy / K46",
        "units": {"modulus": "MPa", "density": "g/cm^3", "length": "micrometre",
                  "volume_fraction": "dimensionless (particles incl. hollow cores)"},
        "constituents": {
            "matrix": {"name": matrix.name, "E_mpa": matrix.E, "nu": matrix.nu, "rho_g_cc": matrix.rho},
            "shell": {"name": part.shell.name, "E_mpa": part.shell.E, "nu": part.shell.nu,
                      "rho_g_cc": part.shell.rho},
            "particle": {"grade": "K46", "eta": part.eta, "true_density_g_cc": part.true_density,
                         "diameter_um": part.diameter_um},
            "equivalent_particle": {"E_mpa": eq.E, "nu": eq.nu, "K_mpa": eq.K, "G_mpa": eq.G},
        },
        "rve": {"target_vf": VF_TARGET, "realised_vf": vf_used, "n_spheres": N_SPHERES,
                "resolutions": list(RESOLUTIONS), "seeds": list(SEEDS),
                "bc": "KUBC", "fe_mode": "equivalent", "packing": "periodic RSA (foamsim.rve)"},
        "self_check_homogeneous_box": hom,
        "fe_runs": runs,
        "fe_summary": {"n_runs": len(runs), "E_mean_mpa": E_mean, "E_std_mpa": E_sd,
                       "E_min_mpa": E_min, "E_max_mpa": E_max,
                       "E_spread_pct_of_mean": 100.0 * (E_max - E_min) / E_mean,
                       "rho_g_cc": runs[0]["rho_g_cc"]},
        "analytical": {
            "mori_tanaka": {"model": mt.model, "E_mpa": mt.E, "K_mpa": mt.K, "G_mpa": mt.G,
                            "nu": mt.nu, "rho_g_cc": mt.rho},
            "differential": {"model": ds.model, "E_mpa": ds.E, "nu": ds.nu},
            "hashin_shtrikman": hs,
        },
        "comparison": {
            "rel_diff_mean_fe_vs_mt": rel_mean_vs_mt,
            "rel_diff_mean_fe_vs_mt_pct": 100.0 * rel_mean_vs_mt,
            "fe_inside_hs_bounds_all_runs": all_inside,
            "fe_mean_inside_hs_bounds": mean_inside,
            "mt_tolerance": MT_TOL,
        },
        "checks_passed": {
            "homogeneous_box": hom["passed"],
            "fe_inside_hs_bounds": all_inside and mean_inside,
            "fe_vs_mt_within_25pct": mt_check,
            "all": bool(hom["passed"] and all_inside and mean_inside and mt_check),
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    with open("results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # results.csv: per-run table, then the summary/analytical block
    lines = ["section,resolution_n,seed,realised_vf,model,E_mpa,K_mpa,G_mpa,nu,rho_g_cc,"
             "E_reference_mpa,rel_diff_vs_mt,inside_hs_bounds"]
    for r in runs:
        lines.append(f"fe_run,{r['resolution_n']},{r['seed']},{r['realised_vf']:.6f},{r['model']},"
                     f"{r['E_mpa']:.4f},{r['K_mpa']:.4f},{r['G_mpa']:.4f},{r['nu']:.5f},"
                     f"{r['rho_g_cc']:.5f},{mt.E:.4f},{r['rel_diff_vs_mt']:.5f},{r['inside_hs']}")
    lines.append(f"fe_mean,,,{vf_used:.6f},FE-KUBC-equivalent-mean,{E_mean:.4f},,,,"
                 f"{runs[0]['rho_g_cc']:.5f},{mt.E:.4f},{rel_mean_vs_mt:.5f},{mean_inside}")
    lines.append(f"fe_std,,,{vf_used:.6f},FE-KUBC-equivalent-std,{E_sd:.4f},,,,,,,")
    lines.append(f"fe_range,,,{vf_used:.6f},FE-KUBC-equivalent-minmax,{E_min:.4f},,,,,{E_max:.4f},,")
    lines.append(f"analytical,,,{vf_used:.6f},{mt.model},{mt.E:.4f},{mt.K:.4f},{mt.G:.4f},"
                 f"{mt.nu:.5f},{mt.rho:.5f},,,")
    lines.append(f"analytical,,,{vf_used:.6f},{ds.model},{ds.E:.4f},,,{ds.nu:.5f},,,,")
    lines.append(f"bound,,,{vf_used:.6f},HS_lower,{hs['E_lo']:.4f},{hs['K_lo']:.4f},{hs['G_lo']:.4f},,,,,")
    lines.append(f"bound,,,{vf_used:.6f},HS_upper,{hs['E_hi']:.4f},{hs['K_hi']:.4f},{hs['G_hi']:.4f},,,,,")
    lines.append(f"self_check,{hom['n']},,0.000000,FE-homogeneous-box,{hom['E_fe_mpa']:.6f},,,"
                 f"{hom['nu_fe']:.6f},,{hom['E_matrix_mpa']:.6f},{hom['rel_diff_E']:.3e},{hom['passed']}")
    with open("results.csv", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---------------------------------------------------------------- figure
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax.axhspan(hs["E_lo"], hs["E_hi"], color="0.85", label="HS bounds")
    ax.axhline(hs["E_lo"], color="0.5", lw=0.8)
    ax.axhline(hs["E_hi"], color="0.5", lw=0.8)
    ax.axhline(mt.E, color="C3", lw=1.8, label=f"Mori-Tanaka ({mt.E:.0f} MPa)")
    ax.axhline(ds.E, color="C1", lw=1.2, ls="--", label=f"Differential ({ds.E:.0f} MPa)")
    ax.axhline(matrix.E, color="k", lw=1.0, ls=":", label=f"matrix E ({matrix.E:.0f} MPa)")
    xs = np.arange(len(runs))
    ax.plot(xs, E_fe, "o", color="C0", ms=9, label="FE (KUBC)")
    ax.errorbar([len(runs) + 0.6], [E_mean], yerr=[E_sd], fmt="s", color="C0", ms=10, capsize=5,
                label=f"FE mean $\\pm$ sd ({E_mean:.0f} $\\pm$ {E_sd:.0f} MPa)")
    ax.set_xticks(list(xs) + [len(runs) + 0.6])
    ax.set_xticklabels([f"n={r['resolution_n']}\nseed {r['seed']}" for r in runs] + ["mean"])
    ax.set_ylabel("Effective Young's modulus $E$ (MPa)")
    ax.set_title(f"epoxy / K46, vf = {vf_used:.3f}")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3, axis="y")

    # mesh-convergence view: E vs resolution, one line per seed
    for seed, c in zip(SEEDS, ["C0", "C4"]):
        sel = [r for r in runs if r["seed"] == seed]
        ax2.plot([r["resolution_n"] for r in sel], [r["E_mpa"] for r in sel], "o-", color=c,
                 label=f"seed {seed}")
    ax2.axhline(mt.E, color="C3", lw=1.8, label="Mori-Tanaka")
    ax2.axhspan(hs["E_lo"], hs["E_hi"], color="0.85", zorder=0)
    ax2.set_xlabel("voxel resolution $n$ (elements per box edge)")
    ax2.set_ylabel("$E$ (MPa)")
    ax2.set_title("mesh sensitivity (KUBC stiffens as mesh coarsens)")
    ax2.set_xticks(list(RESOLUTIONS))
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.suptitle("FE RVE homogenization vs analytical estimates (units: MPa)", fontsize=11)
    fig.tight_layout()
    fig.savefig("fe_vs_analytical.png", dpi=160)

    # ---------------------------------------------------------------- summary
    print("\n" + "-" * 78)
    print("TABLE  FE modulus per (resolution, seed)   [MPa]")
    print(f"{'n':>4} {'seed':>5} {'vf':>7} {'E_FE':>10} {'nu':>7} {'rel. diff vs MT':>17} {'in HS':>7}")
    for r in runs:
        print(f"{r['resolution_n']:>4} {r['seed']:>5} {r['realised_vf']:>7.4f} {r['E_mpa']:>10.1f} "
              f"{r['nu']:>7.4f} {100 * r['rel_diff_vs_mt']:>16.1f}% {str(r['inside_hs']):>7}")
    print(f"  FE mean = {E_mean:.1f} MPa, sd = {E_sd:.1f} MPa, range = [{E_min:.1f}, {E_max:.1f}] MPa "
          f"({100 * (E_max - E_min) / E_mean:.1f}% of mean)")
    print(f"  Mori-Tanaka       E = {mt.E:.1f} MPa   (rho = {mt.rho:.3f} g/cm3)")
    print(f"  Differential      E = {ds.E:.1f} MPa")
    print(f"  HS bounds         E in [{hs['E_lo']:.1f}, {hs['E_hi']:.1f}] MPa")
    print(f"  relative difference (FE mean - MT)/MT = {100 * rel_mean_vs_mt:+.1f}%")

    print("\nPHYSICS SANITY SUMMARY")
    print(f"  [{'PASS' if hom['passed'] else 'FAIL'}] homogeneous box returns the matrix moduli "
          f"(rel. err {hom['rel_diff_E']:.1e}) - FE machinery is correct.")
    print(f"  [{'PASS' if all_inside and mean_inside else 'FAIL'}] every FE modulus and the FE mean lie "
          f"inside the Hashin-Shtrikman bounds [{hs['E_lo']:.0f}, {hs['E_hi']:.0f}] MPa.")
    print(f"  [{'PASS' if mt_check else 'FAIL'}] |FE mean - MT| / MT = {100 * abs(rel_mean_vs_mt):.1f}% "
          f"< {100 * MT_TOL:.0f}%.")
    stiffer = "above" if E_mean > mt.E else "below"
    print(f"  K46 microballoons (equivalent E = {eq.E:.0f} MPa) are stiffer than the epoxy "
          f"({matrix.E:.0f} MPa), so E rises above the matrix while density falls to "
          f"{mt.rho:.3f} g/cm3 (matrix {matrix.rho:.2f}) - the expected syntactic-foam trade-off.")
    print(f"  FE (KUBC) sits {stiffer} Mori-Tanaka, consistent with KUBC being an upper-bound-type "
          f"estimate on a finite {N_SPHERES}-sphere RVE; the {len(runs)}-run spread "
          f"({100 * (E_max - E_min) / E_mean:.1f}% of the mean) is the mesh+realisation uncertainty, "
          f"not a physical one.")
    print(f"  ALL CHECKS {'PASSED' if results['checks_passed']['all'] else 'DID NOT PASS'}.")
    print("  Wrote results.json, results.csv, fe_vs_analytical.png")


if __name__ == "__main__":
    main()
