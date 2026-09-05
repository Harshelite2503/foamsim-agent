"""Inverse design: lightest epoxy / glass-microballoon syntactic foam with E_c >= 2500 MPa.

Design variables : wall ratio eta in [0.80, 0.97]  (equivalently a 3M glass-bubble grade)
                   particle volume fraction vf in [0, 0.60]
Objective        : minimise composite density (g/cm^3)
Constraint       : compressive (Young's) modulus >= 2500 MPa

Model: analytical micromechanics from the foamsim toolkit.
  primary   HP-MT  (hollow-sphere equivalent particle -> Mori-Tanaka)
  check     HP-DS  (same equivalent particle -> differential scheme)
  bounds    Hashin-Shtrikman bounds for matrix + equivalent particle.

Outputs: printed report, sweep_grid.csv, pareto_density_vs_modulus.csv, tradeoff_density_vs_modulus.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foamsim import MATERIALS, hollow_particle
from foamsim.materials import HollowParticle
from foamsim.micromechanics import (RCP, density, hashin_shtrikman_bounds, hollow_particle_differential,
                                    hollow_particle_mori_tanaka, hollow_sphere_equivalent)

E_TARGET = 2500.0        # MPa, compressive modulus requirement
VF_MAX = 0.60            # imposed by the task
ETA_LO, ETA_HI = 0.80, 0.97
MATRIX = MATERIALS["epoxy"]
GLASS = MATERIALS["glass"]
GRADES = ["K1", "K15", "K20", "K25", "S22", "S32", "S38", "K46", "S60"]


# ----------------------------------------------------------------------------- premise / sanity
def premise_checks() -> None:
    print("=" * 78)
    print("0. PREMISE AND SELF-CHECKS")
    print("=" * 78)
    print(f"matrix        : {MATRIX.name}  E={MATRIX.E:.0f} MPa  nu={MATRIX.nu}  rho={MATRIX.rho} g/cm^3")
    print(f"shell         : {GLASS.name}  E={GLASS.E:.0f} MPa  nu={GLASS.nu}  rho={GLASS.rho} g/cm^3")
    print(f"vf_max={VF_MAX} vs random close packing RCP={RCP}  -> "
          f"{'OK (monodisperse-realisable)' if VF_MAX <= RCP else 'NOT realisable'}")
    print(f"eta window [{ETA_LO}, {ETA_HI}] is inside [0,1)  -> OK (eta>=1 would be impossible)")

    # vf = 0 must return the matrix
    e0 = hollow_particle_mori_tanaka(MATRIX, HollowParticle(GLASS, 0.9), vf=0.0)
    assert abs(e0.E - MATRIX.E) < 1e-6 * MATRIX.E and abs(e0.rho - MATRIX.rho) < 1e-9
    print(f"vf=0 limit    : E={e0.E:.1f} MPa, rho={e0.rho:.3f} -> reproduces the neat matrix  OK")

    # every estimate must lie inside the HS band
    worst = 0.0
    for eta in np.linspace(ETA_LO, ETA_HI, 9):
        p = HollowParticle(GLASS, float(eta))
        for vf in np.linspace(0.05, VF_MAX, 12):
            b = hashin_shtrikman_bounds(MATRIX, p, float(vf))
            for est in (hollow_particle_mori_tanaka(MATRIX, p, float(vf)).E,
                        hollow_particle_differential(MATRIX, p, float(vf)).E):
                assert b["E_lo"] - 1e-6 <= est <= b["E_hi"] + 1e-6, (eta, vf, est, b)
                worst = max(worst, (b["E_hi"] - b["E_lo"]) / est)
    print(f"HS containment: all HP-MT and HP-DS estimates inside HS bounds  OK "
          f"(widest band = {100*worst:.0f}% of the estimate)")

    # is the target reachable at all? -> best case is the stiffest particle at max vf
    best_hi = max(hashin_shtrikman_bounds(MATRIX, HollowParticle(GLASS, float(eta)), VF_MAX)["E_hi"]
                  for eta in np.linspace(ETA_LO, ETA_HI, 200))
    print(f"feasibility   : max HS upper bound over the design box = {best_hi:.0f} MPa "
          f"vs target {E_TARGET:.0f} MPa -> "
          f"{'target is NOT above the bound, so it is not ruled out' if E_TARGET <= best_hi else 'IMPOSSIBLE'}")
    print(f"note          : the neat matrix already has E={MATRIX.E:.0f} MPa >= {E_TARGET:.0f} MPa, "
          "so vf=0 (rho=1.180) is a trivially feasible point; the design problem is how much density\n"
          "                can be removed before the modulus drops through the target.\n")


# ----------------------------------------------------------------------------- sweep
def sweep(n_eta: int = 69, n_vf: int = 121) -> pd.DataFrame:
    rows = []
    for eta in np.linspace(ETA_LO, ETA_HI, n_eta):
        p = HollowParticle(GLASS, float(eta))
        for vf in np.linspace(0.0, VF_MAX, n_vf):
            vf = float(vf)
            mt = hollow_particle_mori_tanaka(MATRIX, p, vf)
            ds = hollow_particle_differential(MATRIX, p, vf)
            b = hashin_shtrikman_bounds(MATRIX, p, vf)
            rows.append({"eta": float(eta), "vf": vf,
                         "rho_g_cc": mt.rho,
                         "particle_true_density_g_cc": p.true_density,
                         "E_mt_mpa": mt.E, "E_ds_mpa": ds.E,
                         "E_hs_lo_mpa": b["E_lo"], "E_hs_hi_mpa": b["E_hi"],
                         "nu_mt": mt.nu, "specific_E": mt.E / mt.rho})
    return pd.DataFrame(rows)


def optimum(df: pd.DataFrame, col: str) -> pd.Series:
    """Lightest point in the grid whose modulus (column `col`) meets the target."""
    feas = df[df[col] >= E_TARGET]
    return feas.loc[feas["rho_g_cc"].idxmin()]


def refine(col_model, eta_seed: float, vf_seed: float) -> tuple[float, float]:
    """Local refinement of (eta, vf) around a grid optimum by nested bisection.

    For a fixed eta the modulus is monotone in vf over this box, so the binding design is the
    largest feasible vf; density then decreases with vf and with eta, so we scan eta finely and
    bisect vf for each."""
    best = None
    for eta in np.linspace(max(ETA_LO, eta_seed - 0.03), min(ETA_HI, eta_seed + 0.03), 121):
        p = HollowParticle(GLASS, float(eta))
        lo, hi = 0.0, VF_MAX
        if col_model(MATRIX, p, VF_MAX).E >= E_TARGET:
            vf = VF_MAX
        else:
            if col_model(MATRIX, p, 0.0).E < E_TARGET:
                continue
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if col_model(MATRIX, p, mid).E >= E_TARGET:
                    lo = mid
                else:
                    hi = mid
            vf = lo
        rho = density(MATRIX, p, vf)
        if best is None or rho < best[2]:
            best = (float(eta), vf, rho)
    return best[0], best[1]


def report_point(tag: str, eta: float, vf: float) -> dict:
    p = HollowParticle(GLASS, eta)
    mt = hollow_particle_mori_tanaka(MATRIX, p, vf)
    ds = hollow_particle_differential(MATRIX, p, vf)
    b = hashin_shtrikman_bounds(MATRIX, p, vf)
    eq = hollow_sphere_equivalent(p)
    print(f"{tag}")
    print(f"    eta = {eta:.4f}   (particle true density {p.true_density:.4f} g/cm^3, "
          f"wall t/R = {1-eta:.4f}, equivalent particle E = {eq.E:.0f} MPa)")
    print(f"    vf  = {vf:.4f}")
    print(f"    composite density rho = {mt.rho:.4f} g/cm^3   "
          f"({100*(MATRIX.rho-mt.rho)/MATRIX.rho:.1f}% lighter than neat epoxy)")
    print(f"    E (HP-MT) = {mt.E:.1f} MPa    E (HP-DS) = {ds.E:.1f} MPa    nu = {mt.nu:.3f}")
    print(f"    HS band   = [{b['E_lo']:.1f}, {b['E_hi']:.1f}] MPa -> both estimates inside: "
          f"{b['E_lo']-1e-6 <= mt.E <= b['E_hi']+1e-6 and b['E_lo']-1e-6 <= ds.E <= b['E_hi']+1e-6}")
    print(f"    target {E_TARGET:.0f} MPa vs HS upper bound {b['E_hi']:.1f} MPa -> "
          f"{'feasible (target below the bound)' if E_TARGET <= b['E_hi'] else 'infeasible'}")
    print(f"    specific modulus E/rho = {mt.E/mt.rho:.0f} MPa/(g/cm^3)")
    return {"eta": eta, "vf": vf, "rho": mt.rho, "E_mt": mt.E, "E_ds": ds.E,
            "E_hs_lo": b["E_lo"], "E_hs_hi": b["E_hi"]}


def grade_table() -> pd.DataFrame:
    rows = []
    for g in GRADES:
        p = hollow_particle(g)
        in_window = ETA_LO <= p.eta <= ETA_HI
        # largest feasible vf for this grade (HP-MT), by bisection
        lo, hi = 0.0, VF_MAX
        if hollow_particle_mori_tanaka(MATRIX, p, VF_MAX).E >= E_TARGET:
            vf = VF_MAX
        else:
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if hollow_particle_mori_tanaka(MATRIX, p, mid).E >= E_TARGET:
                    lo = mid
                else:
                    hi = mid
            vf = lo
        mt = hollow_particle_mori_tanaka(MATRIX, p, vf)
        ds = hollow_particle_differential(MATRIX, p, vf)
        b = hashin_shtrikman_bounds(MATRIX, p, vf)
        rows.append({"grade": g, "eta": p.eta, "particle_rho": p.true_density,
                     "in_eta_window": in_window, "vf_max_feasible": vf,
                     "rho_g_cc": mt.rho, "E_mt_mpa": mt.E, "E_ds_mpa": ds.E,
                     "E_hs_lo": b["E_lo"], "E_hs_hi": b["E_hi"]})
    return pd.DataFrame(rows)


def pareto(df: pd.DataFrame, col: str = "E_mt_mpa") -> pd.DataFrame:
    """Minimum achievable density as a function of the required modulus."""
    out = []
    for E_req in np.arange(1000.0, 3001.0, 25.0):
        feas = df[df[col] >= E_req]
        if feas.empty:
            continue
        r = feas.loc[feas["rho_g_cc"].idxmin()]
        out.append({"E_required_mpa": E_req, "min_rho_g_cc": r["rho_g_cc"],
                    "eta": r["eta"], "vf": r["vf"], "E_achieved_mpa": r[col]})
    return pd.DataFrame(out)


def main() -> None:
    premise_checks()

    df = sweep()
    df.to_csv("sweep_grid.csv", index=False)
    print("=" * 78)
    print(f"1. DESIGN SWEEP  ({len(df)} points: eta in [{ETA_LO},{ETA_HI}], vf in [0,{VF_MAX}])  "
          "-> sweep_grid.csv")
    print("=" * 78)
    print(f"    density range  {df.rho_g_cc.min():.3f} - {df.rho_g_cc.max():.3f} g/cm^3")
    print(f"    E (HP-MT) range {df.E_mt_mpa.min():.0f} - {df.E_mt_mpa.max():.0f} MPa")
    n_feas = int((df.E_mt_mpa >= E_TARGET).sum())
    print(f"    feasible points (E_HP-MT >= {E_TARGET:.0f}): {n_feas} / {len(df)}\n")

    print("=" * 78)
    print("2. OPTIMUM (continuous eta)")
    print("=" * 78)
    seed = optimum(df, "E_mt_mpa")
    eta_o, vf_o = refine(hollow_particle_mori_tanaka, float(seed["eta"]), float(seed["vf"]))
    opt = report_point("  >> HP-MT optimum (primary design model):", eta_o, vf_o)

    seed_ds = optimum(df, "E_ds_mpa")
    eta_d, vf_d = refine(hollow_particle_differential, float(seed_ds["eta"]), float(seed_ds["vf"]))
    print()
    report_point("  >> HP-DS optimum (conservative cross-check model):", eta_d, vf_d)

    print("\n" + "=" * 78)
    print("3. BEST 3M GRADE (discrete alternative)")
    print("=" * 78)
    gt = grade_table()
    gt.to_csv("grade_table.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(gt.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    gt_win = gt[gt.in_eta_window & (gt.E_mt_mpa >= E_TARGET - 1e-6)]
    if not gt_win.empty:
        bg = gt_win.loc[gt_win["rho_g_cc"].idxmin()]
        print()
        report_point(f"  >> best grade inside the eta window: {bg['grade']}", float(bg["eta"]),
                     float(bg["vf_max_feasible"]))
    gt_any = gt[gt.E_mt_mpa >= E_TARGET - 1e-6]
    ba = gt_any.loc[gt_any["rho_g_cc"].idxmin()]
    print(f"\n    (over ALL listed grades, ignoring the eta window, the lightest feasible is "
          f"{ba['grade']} at eta={ba['eta']:.4f}, vf={ba['vf_max_feasible']:.4f}, "
          f"rho={ba['rho_g_cc']:.4f} g/cm^3)")

    print("\n" + "=" * 78)
    print("4. TRADE-OFF CURVE: minimum density vs required modulus")
    print("=" * 78)
    pf = pareto(df)
    pf.to_csv("pareto_density_vs_modulus.csv", index=False)
    show = pf[np.isclose(pf.E_required_mpa % 250, 0)]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("  -> pareto_density_vs_modulus.csv")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for eta in [0.80, 0.85, 0.90, 0.93, 0.95, 0.97]:
        s = df[np.isclose(df.eta, df.eta.iloc[(df.eta - eta).abs().argmin()])]
        ax[0].plot(s.rho_g_cc, s.E_mt_mpa, lw=1.2, label=f"eta={s.eta.iloc[0]:.3f}")
    ax[0].axhline(E_TARGET, color="k", ls="--", lw=1, label=f"target {E_TARGET:.0f} MPa")
    ax[0].plot(opt["rho"], opt["E_mt"], "r*", ms=14, label="optimum")
    ax[0].set_xlabel("density (g/cm$^3$)"); ax[0].set_ylabel("E, HP-MT (MPa)")
    ax[0].set_title("Density vs modulus, vf sweep at fixed eta"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

    ax[1].plot(pf.E_required_mpa, pf.min_rho_g_cc, "b-", lw=2)
    ax[1].axvline(E_TARGET, color="k", ls="--", lw=1)
    ax[1].plot(opt["E_mt"], opt["rho"], "r*", ms=14)
    ax[1].set_xlabel("required modulus (MPa)"); ax[1].set_ylabel("minimum achievable density (g/cm$^3$)")
    ax[1].set_title("Pareto front (HP-MT)"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig("tradeoff_density_vs_modulus.png", dpi=150)
    print("  -> tradeoff_density_vs_modulus.png")

    print("\n" + "=" * 78)
    print("5. ANSWER")
    print("=" * 78)
    print(f"  lightest design meeting E >= {E_TARGET:.0f} MPa (HP-MT):")
    print(f"    eta = {opt['eta']:.4f} (t/R = {1-opt['eta']:.4f}); vf = {opt['vf']:.4f}; "
          f"rho = {opt['rho']:.4f} g/cm^3; E = {opt['E_mt']:.1f} MPa")
    print(f"    HS band at that point [{opt['E_hs_lo']:.0f}, {opt['E_hs_hi']:.0f}] MPa -> "
          "estimate inside, target below the upper bound: feasible.")
    print("  caveat: HP-MT is an idealised estimate; HP-DS is lower, and measured syntactic-foam moduli\n"
          "          typically fall 20-40% below HP-MT (matrix porosity, particle breakage, imperfect\n"
          "          bonding). Treat the optimum as an upper-bound-of-performance design point.")


if __name__ == "__main__":
    main()
