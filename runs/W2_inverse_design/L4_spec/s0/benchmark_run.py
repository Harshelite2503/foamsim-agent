"""W2 inverse design: lightest epoxy / glass-microballoon syntactic foam with E >= 3500 MPa.

Model
-----
Analytical micromechanics from the `foamsim` toolkit:
  * each hollow glass microballoon (wall ratio eta = r_in/r_out) is replaced by an equivalent
    homogeneous solid sphere -- K_p exact (Hashin 1962 composite-sphere assemblage with a void
    core), G_p the Hashin-Shtrikman upper bound of the porous shell;
  * the equivalent particles are homogenized into the epoxy matrix with Mori-Tanaka
    (Benveniste 1987)  -> HP-MT, the primary model here;
  * the differential scheme (McLaughlin 1977) -> HP-DS is run in parallel as a model-spread
    estimate (two defensible schemes for the same microstructure);
  * Hashin-Shtrikman bounds for the same two-phase system give the feasibility check:
    no microstructure of these constituents can exceed E_hi at a given vf.

Assumptions: linear elasticity, isotropy, perfectly bonded spherical particles, monodisperse
non-interacting hollow spheres, no matrix porosity, no particle breakage, small strain.
Compressive modulus is identified with the effective Young's modulus (the usual convention for
syntactic foams).

Units: moduli MPa, density g/cm^3, diameters micrometres; vf = particle volume fraction
INCLUDING the hollow cores (dimensionless).

Outputs (written into the current directory): results.json, results.csv, tradeoff.png
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
)

# ----------------------------------------------------------------------------- design space
TARGET_E = 3500.0        # MPa, required compressive modulus
ETA_MIN, ETA_MAX = 0.80, 0.97
VF_MAX = 0.60            # requested cap; hard physical cap is RCP = 0.64
GRADES = ["K1", "K15", "K20", "K25", "S22", "S32", "S38", "K46", "S60"]
N_ETA, N_VF = 69, 121    # eta step 0.0025, vf step 0.005


def evaluate(matrix, particle, vf) -> dict:
    """One design point: both models + HS bounds + density."""
    mt = hollow_particle_mori_tanaka(matrix, particle, vf)
    ds = hollow_particle_differential(matrix, particle, vf)
    hs = hashin_shtrikman_bounds(matrix, particle, vf)
    return {
        "eta": particle.eta,
        "particle_true_density_g_cc": particle.true_density,
        "vf": vf,
        "rho_g_cc": mt.rho,
        "E_mt_mpa": mt.E,
        "E_ds_mpa": ds.E,
        "nu_mt": mt.nu,
        "E_hs_lo_mpa": hs["E_lo"],
        "E_hs_hi_mpa": hs["E_hi"],
        "inside_hs_mt": bool(hs["E_lo"] - 1e-6 <= mt.E <= hs["E_hi"] + 1e-6),
        "inside_hs_ds": bool(hs["E_lo"] - 1e-6 <= ds.E <= hs["E_hi"] + 1e-6),
        "specific_E_mpa_cc_g": mt.E / mt.rho,
    }


def sweep(matrix, etas, vfs) -> pd.DataFrame:
    rows = []
    for eta in etas:
        p = HollowParticle(MATERIALS["glass"], eta=float(eta))
        for vf in vfs:
            rows.append(evaluate(matrix, p, float(vf)))
    return pd.DataFrame(rows)


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """For each achievable modulus level, the minimum density that reaches it (HP-MT)."""
    d = df.sort_values("E_mt_mpa", ascending=False).reset_index(drop=True)
    best, out = np.inf, []
    for _, r in d.iterrows():          # descending modulus: keep points that lower the density
        if r["rho_g_cc"] < best - 1e-12:
            best = r["rho_g_cc"]
            out.append(r)
    return pd.DataFrame(out).sort_values("E_mt_mpa").reset_index(drop=True)


def main() -> None:
    matrix = MATERIALS["epoxy"]
    glass = MATERIALS["glass"]

    # ---------------------------------------------------------------- 1. self-checks
    p_ref = HollowParticle(glass, eta=0.90)
    e0 = hollow_particle_mori_tanaka(matrix, p_ref, 0.0)
    d0 = hollow_particle_differential(matrix, p_ref, 0.0)
    rho0 = density(matrix, p_ref, 0.0)
    checks = {
        "vf0_E_mt_mpa": e0.E,
        "vf0_E_ds_mpa": d0.E,
        "vf0_rho_g_cc": rho0,
        "vf0_recovers_matrix_E": bool(abs(e0.E - matrix.E) < 1e-6 * matrix.E
                                      and abs(d0.E - matrix.E) < 1e-6 * matrix.E),
        "vf0_recovers_matrix_rho": bool(abs(rho0 - matrix.rho) < 1e-9),
        "matrix_alone_infeasible": bool(matrix.E < TARGET_E),
        "packing_limit_rcp": RCP,
        "vf_cap_respected": bool(VF_MAX <= RCP),
    }
    # solid-sphere limit: eta -> 0 must give plain Mori-Tanaka with solid glass spheres
    solid = HollowParticle(glass, eta=0.0)
    e_solid = hollow_particle_mori_tanaka(matrix, solid, 0.3)
    hs_solid = hashin_shtrikman_bounds(matrix, solid, 0.3)
    checks["eta0_vf03_E_mpa"] = e_solid.E
    checks["eta0_inside_hs"] = bool(hs_solid["E_lo"] <= e_solid.E <= hs_solid["E_hi"])
    checks["eta0_stiffer_than_matrix"] = bool(e_solid.E > matrix.E)
    if not (checks["vf0_recovers_matrix_E"] and checks["vf0_recovers_matrix_rho"]):
        raise AssertionError("self-check failed: vf=0 must recover the neat matrix")
    if not checks["matrix_alone_infeasible"]:
        raise AssertionError("self-check failed: matrix alone was expected to be below target")

    # ---------------------------------------------------------------- 2. continuous eta sweep
    etas = np.linspace(ETA_MIN, ETA_MAX, N_ETA)
    vfs = np.linspace(0.0, VF_MAX, N_VF)
    df = sweep(matrix, etas, vfs)
    df["feasible_mt"] = df["E_mt_mpa"] >= TARGET_E
    df["feasible_ds"] = df["E_ds_mpa"] >= TARGET_E
    df["feasible_hs_upper"] = df["E_hs_hi_mpa"] >= TARGET_E   # any microstructure could reach it
    if not df["inside_hs_mt"].all() or not df["inside_hs_ds"].all():
        raise AssertionError("consistency check failed: an estimate fell outside the HS bounds")

    feas = df[df["feasible_mt"]]
    if feas.empty:
        raise RuntimeError("no design in the requested box reaches the target modulus")
    opt = feas.loc[feas["rho_g_cc"].idxmin()]

    # robust variant: feasible under BOTH models (the conservative design)
    feas_both = df[df["feasible_mt"] & df["feasible_ds"]]
    opt_rob = feas_both.loc[feas_both["rho_g_cc"].idxmin()] if not feas_both.empty else None

    # ---------------------------------------------------------------- 3. 3M grade sweep
    grade_rows = []
    for g in GRADES:
        p = hollow_particle(g)
        for vf in vfs:
            r = evaluate(matrix, p, float(vf))
            r["grade"] = g
            grade_rows.append(r)
    gdf = pd.DataFrame(grade_rows)
    gdf["feasible_mt"] = gdf["E_mt_mpa"] >= TARGET_E
    gdf["in_eta_window"] = (gdf["eta"] >= ETA_MIN) & (gdf["eta"] <= ETA_MAX)
    gfeas = gdf[gdf["feasible_mt"] & gdf["in_eta_window"]]
    gopt = gfeas.loc[gfeas["rho_g_cc"].idxmin()] if not gfeas.empty else None

    # ---------------------------------------------------------------- 4. trade-off curve
    front = pareto_front(df)

    # minimum density reachable at each eta (the target-touching vf), for the figure
    curves = {}
    for eta in [0.80, 0.85, 0.90, 0.94, 0.97]:
        sub = df[np.isclose(df["eta"], eta, atol=1e-9)].sort_values("vf")
        if not sub.empty:
            curves[f"{eta:.2f}"] = sub

    # ---------------------------------------------------------------- 5. figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for lbl, sub in curves.items():
        ax[0].plot(sub["rho_g_cc"], sub["E_mt_mpa"], lw=1.6, label=f"eta = {lbl}")
    ax[0].plot(front["rho_g_cc"], front["E_mt_mpa"], "k--", lw=2.2,
               label="Pareto front (min density)")
    ax[0].axhline(TARGET_E, color="crimson", ls=":", lw=1.5, label=f"target {TARGET_E:.0f} MPa")
    ax[0].plot(opt["rho_g_cc"], opt["E_mt_mpa"], "r*", ms=16, zorder=5, label="optimum")
    ax[0].set_xlabel("density  (g/cm$^3$)")
    ax[0].set_ylabel("compressive modulus $E$  (MPa)")
    ax[0].set_title("Trade-off: density vs modulus (HP-MT)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    sub_opt = df[np.isclose(df["eta"], opt["eta"], atol=1e-9)].sort_values("vf")
    ax[1].fill_between(sub_opt["vf"], sub_opt["E_hs_lo_mpa"], sub_opt["E_hs_hi_mpa"],
                       color="0.85", label="HS bounds")
    ax[1].plot(sub_opt["vf"], sub_opt["E_mt_mpa"], "b-", lw=2, label="HP-MT")
    ax[1].plot(sub_opt["vf"], sub_opt["E_ds_mpa"], "g-.", lw=2, label="HP-DS")
    ax[1].axhline(TARGET_E, color="crimson", ls=":", lw=1.5, label=f"target {TARGET_E:.0f} MPa")
    ax[1].axvline(opt["vf"], color="r", ls="--", lw=1.2, label=f"optimal vf = {opt['vf']:.3f}")
    ax[1].set_xlabel("particle volume fraction $v_f$")
    ax[1].set_ylabel("compressive modulus $E$  (MPa)")
    ax[1].set_title(f"Optimal wall ratio eta = {opt['eta']:.4f}: models and HS band")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle("Lightest epoxy / glass-microballoon foam with E >= "
                 f"{TARGET_E:.0f} MPa", fontsize=12)
    fig.tight_layout()
    fig.savefig("tradeoff.png", dpi=160)
    plt.close(fig)

    # ---------------------------------------------------------------- 6. tables
    front_out = front.assign(table="pareto_front")
    grade_best = (gdf[gdf["feasible_mt"]].sort_values("rho_g_cc")
                    .groupby("grade", as_index=False).first().assign(table="grade_optimum"))
    eta_best = (df[df["feasible_mt"]].sort_values("rho_g_cc")
                  .groupby("eta", as_index=False).first().assign(table="eta_optimum"))
    cols = ["table", "grade", "eta", "particle_true_density_g_cc", "vf", "rho_g_cc",
            "E_mt_mpa", "E_ds_mpa", "E_hs_lo_mpa", "E_hs_hi_mpa", "specific_E_mpa_cc_g"]
    out = pd.concat([front_out, eta_best, grade_best], ignore_index=True)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out[cols].to_csv("results.csv", index=False)

    # ---------------------------------------------------------------- 7. json
    def pack(r) -> dict | None:
        if r is None:
            return None
        d = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and
                 not isinstance(v, bool) else bool(v) if isinstance(v, (bool, np.bool_)) else v)
             for k, v in r.items() if k in
             ("grade", "eta", "particle_true_density_g_cc", "vf", "rho_g_cc", "E_mt_mpa",
              "E_ds_mpa", "nu_mt", "E_hs_lo_mpa", "E_hs_hi_mpa", "inside_hs_mt",
              "specific_E_mpa_cc_g")}
        return d

    results = {
        "task": "W2 inverse design: minimum-density epoxy/glass-microballoon foam with E >= "
                f"{TARGET_E:.0f} MPa",
        "units": {"modulus": "MPa", "density": "g/cm^3", "diameter": "micrometre",
                  "vf": "dimensionless particle volume fraction including hollow cores",
                  "eta": "r_inner/r_outer wall ratio"},
        "model": {
            "primary": "HP-MT (hollow-sphere equivalent particle + Mori-Tanaka, Benveniste 1987)",
            "spread": "HP-DS (differential scheme, McLaughlin 1977)",
            "bounds": "Hashin-Shtrikman two-phase bounds (matrix + equivalent particle)",
            "assumptions": ["linear elastic isotropic constituents",
                            "perfectly bonded spherical monodisperse hollow particles",
                            "void core, K_p exact (Hashin 1962), G_p = HS upper bound of shell",
                            "no matrix porosity, no particle breakage, small strain",
                            "compressive modulus identified with effective Young's modulus"],
        },
        "constituents": {
            "matrix": {"name": matrix.name, "E_mpa": matrix.E, "nu": matrix.nu,
                       "rho_g_cc": matrix.rho},
            "shell": {"name": glass.name, "E_mpa": glass.E, "nu": glass.nu, "rho_g_cc": glass.rho},
        },
        "design_space": {"eta_min": ETA_MIN, "eta_max": ETA_MAX, "vf_min": 0.0, "vf_max": VF_MAX,
                         "n_eta": N_ETA, "n_vf": N_VF, "grades": GRADES,
                         "packing_limit_rcp": RCP},
        "target_E_mpa": TARGET_E,
        "self_checks": checks,
        "optimum_continuous_eta": pack(opt),
        "optimum_feasible_under_both_models": pack(opt_rob),
        "optimum_3m_grade": pack(gopt),
        "hs_feasibility": {
            "E_hs_upper_at_optimum_mpa": float(opt["E_hs_hi_mpa"]),
            "E_hs_lower_at_optimum_mpa": float(opt["E_hs_lo_mpa"]),
            "target_below_hs_upper": bool(TARGET_E <= opt["E_hs_hi_mpa"]),
            "estimate_inside_hs_bounds": bool(opt["inside_hs_mt"]),
            "max_E_hs_upper_in_design_space_mpa": float(df["E_hs_hi_mpa"].max()),
            "n_points_where_target_exceeds_hs_upper":
                int((~df["feasible_hs_upper"]).sum()),
            "comment": "the target is achievable only where E_hs_hi >= target; HP-MT sits inside "
                       "the band everywhere, so the design is not an artefact of the model choice",
        },
        "model_spread_at_optimum": {
            "E_mt_mpa": float(opt["E_mt_mpa"]), "E_ds_mpa": float(opt["E_ds_mpa"]),
            "abs_diff_mpa": float(abs(opt["E_mt_mpa"] - opt["E_ds_mpa"])),
            "rel_diff_pct": float(100 * abs(opt["E_mt_mpa"] - opt["E_ds_mpa"]) / opt["E_mt_mpa"]),
            "hs_band_width_mpa": float(opt["E_hs_hi_mpa"] - opt["E_hs_lo_mpa"]),
            "density_penalty_of_robust_design_g_cc":
                (float(opt_rob["rho_g_cc"] - opt["rho_g_cc"]) if opt_rob is not None else None),
        },
        "constraints_respected": {
            "vf_le_0.64": bool(opt["vf"] <= RCP),
            "vf_le_requested_0.60": bool(opt["vf"] <= VF_MAX),
            "eta_in_window": bool(ETA_MIN <= opt["eta"] <= ETA_MAX),
            "E_ge_target": bool(opt["E_mt_mpa"] >= TARGET_E),
            "vf_constraint_active_at_optimum": bool(opt["vf"] >= VF_MAX - 1e-9),
        },
        "tradeoff_curve_density_vs_modulus": {
            "rho_g_cc": [float(x) for x in front["rho_g_cc"]],
            "E_mt_mpa": [float(x) for x in front["E_mt_mpa"]],
            "eta": [float(x) for x in front["eta"]],
            "vf": [float(x) for x in front["vf"]],
        },
        "files": {"json": "results.json", "csv": "results.csv", "figure": "tradeoff.png"},
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---------------------------------------------------------------- 8. sanity summary
    print("=" * 78)
    print("PHYSICS SANITY SUMMARY  (units: E in MPa, density in g/cm^3, eta and vf dimensionless)")
    print("=" * 78)
    print(f"Model: HP-MT (equivalent hollow-sphere particle + Mori-Tanaka); HP-DS as spread; "
          f"HS bounds as feasibility envelope.")
    print(f"Self-check  E(vf=0) = {e0.E:.3f} MPa  vs neat matrix {matrix.E:.1f} MPa "
          f"-> {'PASS' if checks['vf0_recovers_matrix_E'] else 'FAIL'}; "
          f"rho(vf=0) = {rho0:.4f} vs {matrix.rho:.4f} "
          f"-> {'PASS' if checks['vf0_recovers_matrix_rho'] else 'FAIL'}")
    print(f"Self-check  matrix alone ({matrix.E:.0f} MPa) < target ({TARGET_E:.0f} MPa) "
          f"-> infeasible without particles: "
          f"{'PASS' if checks['matrix_alone_infeasible'] else 'FAIL'}")
    print(f"Self-check  eta=0 (solid glass spheres), vf=0.30: E = {e_solid.E:.1f} MPa > matrix, "
          f"inside HS bounds: {checks['eta0_inside_hs']}")
    print(f"Self-check  all {len(df)} sweep points lie inside their HS bounds: True")
    print("-" * 78)
    print(f"OPTIMUM (continuous eta): eta = {opt['eta']:.4f}  "
          f"(particle true density {opt['particle_true_density_g_cc']:.4f} g/cm^3), "
          f"vf = {opt['vf']:.4f}")
    print(f"  density  = {opt['rho_g_cc']:.4f} g/cm^3   "
          f"({100 * (1 - opt['rho_g_cc'] / matrix.rho):.1f} % lighter than neat epoxy "
          f"{matrix.rho:.2f} g/cm^3)")
    print(f"  E (HP-MT) = {opt['E_mt_mpa']:.1f} MPa   E (HP-DS) = {opt['E_ds_mpa']:.1f} MPa   "
          f"spread = {100 * abs(opt['E_mt_mpa'] - opt['E_ds_mpa']) / opt['E_mt_mpa']:.1f} %")
    print(f"  HS band  = [{opt['E_hs_lo_mpa']:.1f}, {opt['E_hs_hi_mpa']:.1f}] MPa; "
          f"target {TARGET_E:.0f} <= HS upper: "
          f"{'PASS (feasible)' if TARGET_E <= opt['E_hs_hi_mpa'] else 'FAIL (infeasible)'}")
    print(f"  specific modulus = {opt['specific_E_mpa_cc_g']:.0f} MPa/(g/cm^3) "
          f"vs {matrix.E / matrix.rho:.0f} for neat epoxy")
    if opt_rob is not None:
        print(f"CONSERVATIVE (E >= target under BOTH HP-MT and HP-DS): eta = {opt_rob['eta']:.4f}, "
              f"vf = {opt_rob['vf']:.4f}, rho = {opt_rob['rho_g_cc']:.4f} g/cm^3 "
              f"(+{opt_rob['rho_g_cc'] - opt['rho_g_cc']:.4f} g/cm^3)")
    if gopt is not None:
        print(f"BEST 3M GRADE in the eta window: {gopt['grade']} (eta = {gopt['eta']:.4f}, true "
              f"density {gopt['particle_true_density_g_cc']:.3f}), vf = {gopt['vf']:.4f}, "
              f"rho = {gopt['rho_g_cc']:.4f} g/cm^3, E = {gopt['E_mt_mpa']:.1f} MPa")
    print(f"Constraints: vf = {opt['vf']:.3f} <= 0.60 requested <= RCP {RCP} "
          f"(monodisperse packing limit): PASS")
    if opt["vf"] >= VF_MAX - 1e-9:
        print(f"NOTE: the optimum sits ON the vf = {VF_MAX} boundary -- the constraint is active, "
              f"so lighter designs exist above it (physical ceiling is RCP = {RCP}, and vf > ~0.55 "
              f"already needs polydisperse packing).")
    print("Caveat: analytical estimate for an idealised microstructure. Measured syntactic-foam "
          "moduli typically fall 20-40 % below HP-MT because of matrix porosity, imperfect "
          "bonding and particle breakage; treat E as an upper-ish estimate and keep margin.")
    print("Wrote results.json, results.csv, tradeoff.png")


if __name__ == "__main__":
    main()
