"""Requested task: modulus + density of epoxy / K46 syntactic foam at
particle volume fraction vf = 0.75 (monodisperse) and wall ratio eta = 1.02.

Outcome of the premise check: the requested operating point is NOT physically
admissible, on two independent counts, so no modulus/density is reported at it.
This script

  1. runs the required self-check (E(vf=0) must equal the matrix modulus, 3000 MPa),
  2. checks every input for physical admissibility BEFORE computing anything,
  3. documents the two violations (with the toolkit's own errors as evidence),
  4. reports instead the admissible envelope for epoxy/K46 (vf = 0 ... RCP = 0.64,
     eta taken from the K46 datasheet true density) with an HP-MT / HP-DS spread
     and Hashin-Shtrikman bounds,

and writes results.json, results.csv and modulus_vs_vf.png into the CWD.

Units: E, K, G in MPa; density in g/cm^3; diameters in micrometres; vf, eta, nu
dimensionless. No experimental value is hard-coded anywhere.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from foamsim import MATERIALS, hollow_particle
from foamsim.materials import HollowParticle
from foamsim.micromechanics import (
    RCP,
    density,
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
    hollow_sphere_equivalent,
)

# ----------------------------------------------------------------------------- requested inputs
VF_REQUESTED = 0.75
ETA_REQUESTED = 1.02
FCC = np.pi / (3 * np.sqrt(2))  # 0.7405, densest possible packing of equal spheres (Kepler/Hales)


def self_check(matrix, particle) -> dict:
    """Independent known result: at vf = 0 the composite must be the pure matrix."""
    e0_mt = hollow_particle_mori_tanaka(matrix, particle, vf=0.0)
    e0_ds = hollow_particle_differential(matrix, particle, vf=0.0)
    rho0 = density(matrix, particle, vf=0.0)
    checks = {
        "E(vf=0)_HP-MT_mpa": e0_mt.E,
        "E(vf=0)_HP-DS_mpa": e0_ds.E,
        "E_matrix_mpa": matrix.E,
        "nu(vf=0)_HP-MT": e0_mt.nu,
        "nu_matrix": matrix.nu,
        "rho(vf=0)_g_cc": rho0,
        "rho_matrix_g_cc": matrix.rho,
        "passed": bool(
            abs(e0_mt.E - matrix.E) < 1e-6 * matrix.E
            and abs(e0_ds.E - matrix.E) < 1e-6 * matrix.E
            and abs(e0_mt.nu - matrix.nu) < 1e-9
            and abs(rho0 - matrix.rho) < 1e-12
        ),
    }
    return checks


def premise_checks(matrix) -> dict:
    """Admissibility of the two requested inputs, each probed against the toolkit."""
    violations = []

    # (1) wall ratio eta = r_inner / r_outer must satisfy 0 <= eta < 1.
    try:
        HollowParticle(MATERIALS["glass"], eta=ETA_REQUESTED)
        eta_err = None
        eta_ok = True
    except ValueError as exc:
        eta_err = f"{type(exc).__name__}: {exc}"
        eta_ok = False
    if not eta_ok:
        violations.append(
            f"eta = {ETA_REQUESTED} is not a wall ratio: eta = r_inner/r_outer must lie in [0,1). "
            "eta >= 1 means the cavity is larger than the particle, i.e. negative wall volume and "
            "negative particle mass. If 1.02 was meant as an outer/inner radius ratio, the "
            "corresponding eta would be 1/1.02 = "
            f"{1.0 / ETA_REQUESTED:.4f}, a shell of ~2% wall thickness that no commercial "
            "microballoon has - and it is in any case inconsistent with the K46 grade, whose "
            "datasheet true density fixes eta."
        )

    # (2) volume fraction of monodisperse spheres.
    k46 = hollow_particle("K46")
    try:
        hollow_particle_mori_tanaka(matrix, k46, vf=VF_REQUESTED)
        vf_err = None
        vf_ok = True
    except ValueError as exc:
        vf_err = f"{type(exc).__name__}: {exc}"
        vf_ok = False
    if not vf_ok:
        violations.append(
            f"vf = {VF_REQUESTED} is not realisable for MONODISPERSE spheres: random close packing "
            f"is {RCP:.2f} and even the densest ordered (FCC/HCP) packing of equal spheres is "
            f"{FCC:.4f} < {VF_REQUESTED}. Above ~0.55 a real syntactic foam needs a polydisperse "
            "balloon size distribution (or deliberate particle deformation/crushing), which "
            "contradicts the stated 'monodisperse'."
        )

    return {
        "eta_requested": ETA_REQUESTED,
        "eta_admissible": eta_ok,
        "eta_toolkit_error": eta_err,
        "vf_requested": VF_REQUESTED,
        "vf_admissible": vf_ok,
        "vf_toolkit_error": vf_err,
        "rcp_monodisperse": RCP,
        "fcc_max_monodisperse": float(FCC),
        "all_inputs_admissible": bool(eta_ok and vf_ok),
        "violations": violations,
    }


def admissible_envelope(matrix, particle) -> pd.DataFrame:
    """HP-MT, HP-DS and HS bounds over the admissible vf range, 0 .. RCP."""
    rows = []
    for vf in np.linspace(0.0, RCP, 33):
        mt = hollow_particle_mori_tanaka(matrix, particle, vf)
        ds = hollow_particle_differential(matrix, particle, vf)
        b = hashin_shtrikman_bounds(matrix, particle, vf)
        rows.append(
            {
                "vf": float(vf),
                "E_HP_MT_mpa": mt.E,
                "E_HP_DS_mpa": ds.E,
                "E_HS_lo_mpa": b["E_lo"],
                "E_HS_hi_mpa": b["E_hi"],
                "nu_HP_MT": mt.nu,
                "rho_g_cc": mt.rho,
                "specific_E_mpa_cc_g": mt.E / mt.rho,
                "inside_HS_bounds": bool(
                    b["E_lo"] - 1e-6 <= mt.E <= b["E_hi"] + 1e-6
                    and b["E_lo"] - 1e-6 <= ds.E <= b["E_hi"] + 1e-6
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    matrix = MATERIALS["epoxy"]
    k46 = hollow_particle("K46")  # eta inferred from the 0.46 g/cm^3 datasheet true density
    eq = hollow_sphere_equivalent(k46)

    checks = self_check(matrix, k46)
    if not checks["passed"]:
        raise SystemExit("self-check failed: E(vf=0) does not recover the matrix modulus")

    premise = premise_checks(matrix)

    df = admissible_envelope(matrix, k46)
    df.to_csv("results.csv", index=False)

    # Closest admissible surrogate: the packing ceiling for monodisperse spheres.
    ceiling = df.iloc[-1]
    mt_hi = hollow_particle_mori_tanaka(matrix, k46, RCP)
    ds_hi = hollow_particle_differential(matrix, k46, RCP)
    spread = abs(mt_hi.E - ds_hi.E)
    e_mid = 0.5 * (mt_hi.E + ds_hi.E)

    results = {
        "task": "modulus and density of epoxy/K46 syntactic foam at vf=0.75 (monodisperse), eta=1.02",
        "answered_as_requested": False,
        "refusal_reason": (
            "The requested operating point is physically inadmissible on two independent counts "
            "(eta >= 1; vf above the maximum packing fraction of monodisperse spheres). Reporting a "
            "modulus and density there would be a number without a microstructure behind it, so the "
            "admissible envelope is reported instead."
        ),
        "units": {"E": "MPa", "K": "MPa", "G": "MPa", "density": "g/cm^3",
                  "diameter": "um", "vf": "-", "eta": "-", "nu": "-"},
        "constituents": {
            "matrix": {"name": matrix.name, "E_mpa": matrix.E, "nu": matrix.nu, "rho_g_cc": matrix.rho},
            "particle_grade": "K46",
            "shell": {"name": k46.shell.name, "E_mpa": k46.shell.E, "nu": k46.shell.nu,
                      "rho_g_cc": k46.shell.rho},
            "eta_from_K46_datasheet_density": k46.eta,
            "particle_true_density_g_cc": k46.true_density,
            "equivalent_particle": {"E_mpa": eq.E, "nu": eq.nu, "K_mpa": eq.K, "G_mpa": eq.G},
        },
        "self_check": checks,
        "premise_checks": premise,
        "model": {
            "primary": "HP-MT (hollow_particle_mori_tanaka): Hashin exact hollow-sphere equivalent "
                       "particle + Mori-Tanaka (Benveniste 1987)",
            "cross_check": "HP-DS (differential scheme, McLaughlin 1977)",
            "bounds": "Hashin-Shtrikman bounds for matrix + equivalent particle",
            "assumptions": [
                "linear elasticity, small strain, perfect particle/matrix bonding",
                "spherical, randomly dispersed, non-interacting-to-mean-field particles; isotropic composite",
                "intact balloons (no crushing), no matrix porosity, no interphase",
                "hollow sphere replaced by an equivalent homogeneous particle: K exact (Hashin 1962), "
                "G taken as the HS upper bound of the porous shell (no exact result exists)",
                "the toolkit refuses vf > RCP = 0.64, encoding the monodisperse packing limit",
            ],
        },
        "admissible_envelope": {
            "vf_range": [0.0, RCP],
            "at_packing_ceiling_vf_0.64": {
                "E_HP_MT_mpa": mt_hi.E,
                "E_HP_DS_mpa": ds_hi.E,
                "E_best_estimate_mpa": e_mid,
                "E_model_spread_mpa": spread,
                "E_model_spread_pct": 100 * spread / e_mid,
                "E_HS_lo_mpa": float(ceiling["E_HS_lo_mpa"]),
                "E_HS_hi_mpa": float(ceiling["E_HS_hi_mpa"]),
                "nu_HP_MT": mt_hi.nu,
                "density_g_cc": mt_hi.rho,
            },
            "all_estimates_inside_HS_bounds": bool(df["inside_HS_bounds"].all()),
            "note": "This is the closest admissible point to the request, not the requested point. "
                    "It is an upper limit on vf for monodisperse balloons, and KUBC/FE or experiment "
                    "would typically sit below HP-MT because of matrix porosity and particle breakage.",
        },
        "hypothetical_density_only": {
            "explanation": "Density is a rule of mixtures and is defined for any vf, so for reference "
                           "only: if 0.75 of the volume were K46 balloons at their datasheet true "
                           "density, the mixture density would be the value below. It does not make "
                           "the microstructure realisable and no modulus is quoted with it.",
            "rho_g_cc": VF_REQUESTED * k46.true_density + (1 - VF_REQUESTED) * matrix.rho,
        },
        "what_would_make_the_task_well_posed": [
            "Keep monodisperse balloons and lower vf to <= 0.64 (practically <= ~0.55 for a mixable paste).",
            "Keep vf = 0.75 and specify a polydisperse (bimodal or broad) balloon size distribution, "
            "and state that the packing/mixing route can reach it.",
            "Drop eta = 1.02: either use the K46 grade (eta = %.4f from its true density) or give a "
            "wall ratio in [0,1) / a true density <= the shell density." % k46.eta,
        ],
    }
    with open("results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.fill_between(df["vf"], df["E_HS_lo_mpa"], df["E_HS_hi_mpa"], color="0.85",
                    label="Hashin-Shtrikman bounds")
    ax.plot(df["vf"], df["E_HP_MT_mpa"], "-", lw=2, color="C0", label="HP-MT (primary)")
    ax.plot(df["vf"], df["E_HP_DS_mpa"], "--", lw=2, color="C1", label="HP-DS (cross-check)")
    ax.plot([0.0], [matrix.E], "o", color="k", ms=7, zorder=5,
            label=f"self-check E(vf=0) = {matrix.E:.0f} MPa")
    ax.axvspan(RCP, 0.85, color="C3", alpha=0.10)
    ax.axvline(RCP, color="C3", lw=1.5, label=f"random close packing = {RCP:.2f}")
    ax.axvline(FCC, color="C3", lw=1.2, ls=":", label=f"FCC max (monodisperse) = {FCC:.3f}")
    ax.axvline(VF_REQUESTED, color="k", lw=1.5, ls="-.",
               label=f"requested vf = {VF_REQUESTED} (inadmissible)")
    ax.text(0.695, 0.06, "not realisable\nfor monodisperse\nspheres", color="C3", fontsize=9,
            ha="center", va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("particle volume fraction $v_f$ (-)")
    ax.set_ylabel("Young's modulus $E$ (MPa)")
    ax.set_title("Epoxy / K46 syntactic foam: admissible envelope\n"
                 "requested point ($v_f$=0.75 monodisperse, $\\eta$=1.02) is inadmissible", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("modulus_vs_vf.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------- physics sanity summary
    print("\n" + "=" * 78)
    print("PHYSICS SANITY SUMMARY - epoxy / K46 syntactic foam")
    print("=" * 78)
    print(f"self-check     : E(vf=0) = {checks['E(vf=0)_HP-MT_mpa']:.4f} MPa (HP-MT), "
          f"{checks['E(vf=0)_HP-DS_mpa']:.4f} MPa (HP-DS) vs matrix {matrix.E:.0f} MPa -> PASS; "
          f"rho(vf=0) = {checks['rho(vf=0)_g_cc']:.3f} g/cm3 = matrix")
    print(f"K46 geometry   : eta = {k46.eta:.4f} inferred from the 0.46 g/cm3 datasheet true density; "
          f"equivalent particle E = {eq.E:.0f} MPa (> matrix, so E rises with vf - sign is correct)")
    print("premise checks : REQUESTED POINT IS INADMISSIBLE - no modulus/density is reported at it.")
    for i, v in enumerate(premise["violations"], 1):
        print(f"   ({i}) {v}")
    print(f"   toolkit says : {premise['eta_toolkit_error']}")
    print(f"   toolkit says : {premise['vf_toolkit_error']}")
    print(f"admissible     : vf in [0, {RCP:.2f}], eta = {k46.eta:.4f} (K46). At the packing ceiling "
          f"vf = {RCP:.2f}:")
    print(f"                 E = {e_mid:.0f} MPa (HP-MT {mt_hi.E:.0f} / HP-DS {ds_hi.E:.0f}; "
          f"model spread {spread:.0f} MPa = {100 * spread / e_mid:.1f} %), "
          f"HS band [{ceiling['E_HS_lo_mpa']:.0f}, {ceiling['E_HS_hi_mpa']:.0f}] MPa")
    print(f"                 nu = {mt_hi.nu:.3f}, density = {mt_hi.rho:.3f} g/cm3 "
          f"(matrix {matrix.rho:.2f} g/cm3)")
    print(f"bounds check   : all HP-MT and HP-DS points inside the HS bounds: "
          f"{bool(df['inside_HS_bounds'].all())}")
    print("caveat         : these are mean-field estimates for intact, perfectly bonded balloons; "
          "measured moduli are commonly 20-40 % lower (matrix porosity, balloon breakage).")
    print("wrote          : results.json, results.csv, modulus_vs_vf.png")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
