"""Inverse design: lightest epoxy / glass-microballoon syntactic foam with
compressive (Young's) modulus >= 3500 MPa.

Design variables
    eta = r_inner / r_outer of the glass microballoon (wall-thickness ratio)
    vf  = particle volume fraction (including hollow cores), capped by random
          close packing of monodisperse spheres (RCP = 0.64)

Objective   minimise composite density rho(eta, vf)
Constraint  E_HP-MT(eta, vf) >= 3500 MPa

Models: hollow-particle Mori-Tanaka (HP-MT) as the primary estimate, the
differential scheme (HP-DS) as a cross-check, and Hashin-Shtrikman bounds as
the feasibility / sanity band.
"""
from __future__ import annotations

import numpy as np

from foamsim import MATERIALS, hollow_particle
from foamsim.materials import HollowParticle
from foamsim.micromechanics import (
    RCP,
    density,
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
)

E_TARGET = 3500.0        # MPa
MATRIX = MATERIALS["epoxy"]
SHELL = MATERIALS["glass"]


def sanity_checks() -> None:
    """Known limits before trusting the sweep."""
    p = hollow_particle("K46")
    e0 = hollow_particle_mori_tanaka(MATRIX, p, vf=0.0)
    assert abs(e0.E - MATRIX.E) < 1e-6 * MATRIX.E, "vf=0 must return the matrix modulus"
    print(f"[check] vf=0 -> E = {e0.E:.1f} MPa (matrix E = {MATRIX.E:.1f} MPa)  OK")

    e = hollow_particle_mori_tanaka(MATRIX, p, vf=0.4)
    b = hashin_shtrikman_bounds(MATRIX, p, vf=0.4)
    assert b["E_lo"] <= e.E <= b["E_hi"], "HP-MT must lie inside the HS band"
    print(f"[check] K46 vf=0.40: E = {e.E:.1f} MPa inside HS band "
          f"[{b['E_lo']:.1f}, {b['E_hi']:.1f}] MPa  OK")

    # Feasibility ceiling: stiffest possible microstructure = solid glass spheres at RCP.
    solid = HollowParticle(SHELL, eta=0.0)
    hi = hashin_shtrikman_bounds(MATRIX, solid, vf=RCP)["E_hi"]
    print(f"[check] HS upper bound, solid glass at vf=RCP={RCP}: E_hi = {hi:.1f} MPa "
          f"-> target {E_TARGET:.0f} MPa is {'FEASIBLE' if hi >= E_TARGET else 'INFEASIBLE'}")


def sweep() -> list[dict]:
    """Grid search over (eta, vf); keep every feasible design."""
    etas = np.linspace(0.0, 0.98, 197)         # 0 = solid glass sphere
    vfs = np.linspace(0.0, RCP, 129)
    rows = []
    for eta in etas:
        p = HollowParticle(SHELL, eta=float(eta))
        for vf in vfs:
            e = hollow_particle_mori_tanaka(MATRIX, p, vf=float(vf))
            if e.E >= E_TARGET:
                rows.append({"eta": float(eta), "vf": float(vf), "E_mpa": e.E,
                             "rho_g_cc": e.rho, "true_density_p": p.true_density})
    return rows


def refine(eta: float, rows: list[dict]) -> dict:
    """At the best eta, find the smallest vf meeting the target (bisection)."""
    p = HollowParticle(SHELL, eta=eta)
    lo, hi = 0.0, RCP
    if hollow_particle_mori_tanaka(MATRIX, p, vf=hi).E < E_TARGET:
        raise RuntimeError("target not reachable at this eta")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if hollow_particle_mori_tanaka(MATRIX, p, vf=mid).E >= E_TARGET:
            hi = mid
        else:
            lo = mid
    e = hollow_particle_mori_tanaka(MATRIX, p, vf=hi)
    return {"eta": eta, "vf": hi, "E_mpa": e.E, "rho_g_cc": e.rho}


def main() -> None:
    print("=" * 74)
    print("W2 inverse design: lightest epoxy / glass-microballoon foam, E >= "
          f"{E_TARGET:.0f} MPa")
    print("=" * 74)
    sanity_checks()

    rows = sweep()
    print(f"\n[sweep] {len(rows)} feasible (eta, vf) grid points out of "
          f"{197 * 129} evaluated")
    if not rows:
        print("No feasible design on the grid.")
        return

    best_grid = min(rows, key=lambda r: r["rho_g_cc"])
    print(f"[sweep] grid optimum: eta = {best_grid['eta']:.4f}, vf = {best_grid['vf']:.4f}, "
          f"rho = {best_grid['rho_g_cc']:.4f} g/cm^3")

    # Refine vf at each eta near the grid optimum, then pick the lightest.
    cands = []
    for eta in np.linspace(max(0.0, best_grid["eta"] - 0.05),
                           min(0.98, best_grid["eta"] + 0.05), 101):
        p = HollowParticle(SHELL, eta=float(eta))
        if hollow_particle_mori_tanaka(MATRIX, p, vf=RCP).E < E_TARGET:
            continue
        cands.append(refine(float(eta), rows))
    best = min(cands, key=lambda r: r["rho_g_cc"])

    p = HollowParticle(SHELL, eta=best["eta"])
    ds = hollow_particle_differential(MATRIX, p, vf=best["vf"])
    b = hashin_shtrikman_bounds(MATRIX, p, vf=best["vf"])
    rho_matrix = MATRIX.rho

    print("\n" + "-" * 74)
    print("OPTIMUM (HP-MT)")
    print("-" * 74)
    print(f"  eta (r_in/r_out)      : {best['eta']:.4f}   (wall t/R = {1 - best['eta']:.4f})")
    print(f"  particle true density : {p.true_density:.4f} g/cm^3")
    print(f"  vf (particles)        : {best['vf']:.4f}   (RCP limit {RCP})")
    if best["vf"] > 0.55:
        print("      NOTE: vf > 0.55 is only achievable with a polydisperse size "
              "distribution;\n            monodisperse packing tops out at RCP = 0.64.")
    print(f"  E (HP-MT)             : {best['E_mpa']:.1f} MPa  (target {E_TARGET:.0f})")
    print(f"  E (HP-DS cross-check) : {ds.E:.1f} MPa")
    print(f"  HS band at this point : [{b['E_lo']:.1f}, {b['E_hi']:.1f}] MPa")
    print(f"  density               : {best['rho_g_cc']:.4f} g/cm^3 "
          f"({100 * (1 - best['rho_g_cc'] / rho_matrix):.1f} % lighter than neat epoxy "
          f"at {rho_matrix:.2f} g/cm^3)")
    print(f"  specific modulus      : {best['E_mpa'] / best['rho_g_cc']:.1f} MPa/(g/cm^3)")
    assert b["E_lo"] <= best["E_mpa"] <= b["E_hi"], "optimum must sit inside the HS band"

    # Nearest real 3M grade, for reference.
    print("\nNearest commercial 3M grades (min vf meeting the target, HP-MT):")
    for grade in ["S60", "iM30K", "K46", "iM16K", "H50", "S38", "K25", "K15"]:
        g = hollow_particle(grade)
        if hollow_particle_mori_tanaka(MATRIX, g, vf=RCP).E < E_TARGET:
            print(f"  {grade:6s} eta={g.eta:.3f}  cannot reach {E_TARGET:.0f} MPa below RCP")
            continue
        r = refine(g.eta, rows)
        print(f"  {grade:6s} eta={g.eta:.3f}  vf={r['vf']:.3f}  "
              f"rho={r['rho_g_cc']:.4f} g/cm^3  E={r['E_mpa']:.1f} MPa")

    print("\nCaveat: HP-MT is an idealised estimate (perfect bonding, no matrix porosity,")
    print("no particle breakage); measured syntactic-foam moduli typically fall 20-40 %")
    print("below it, so the design above should be treated as an upper-bound-of-optimism")
    print("target and verified experimentally. The optimum eta is also thinner-walled")
    print("than any standard 3M grade, so the practical choice is the best commercial")
    print("grade in the table above.")


if __name__ == "__main__":
    main()
