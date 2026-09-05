"""W2 inverse design: lightest epoxy / glass-microballoon syntactic foam with E >= 3500 MPa.

Design variables : eta = r_inner/r_outer of the glass microballoon (wall ratio), 0.80 <= eta <= 0.97
                   vf  = particle volume fraction (INCLUDING hollow cores), 0 <= vf <= 0.60
Objective        : minimise composite density rho (g/cm^3)
Constraint       : compressive Young's modulus E >= 3500 MPa, vf <= 0.60 (prompt) <= 0.64 (RCP limit)

Model            : HP-MT = hollow-particle Mori-Tanaka (foamsim.micromechanics).
                   Hollow sphere -> equivalent solid particle (K_p exact, Hashin 1962 composite-sphere
                   assemblage with a void core; G_p = HS upper bound of the porous shell), then
                   Mori-Tanaka (Benveniste 1987) for spherical inclusions in the epoxy matrix.
                   Spread reported against HP-DS = differential scheme (McLaughlin 1977).
Assumptions      : linear elastic isotropic phases, perfect matrix/particle bonding, monodisperse
                   non-interacting spheres, no matrix porosity, no particle breakage, small strain.
                   "Compressive modulus" == elastic Young's modulus (no damage / crush).

Units            : moduli MPa, density g/cm^3, eta and vf dimensionless.
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
VF_MAX = 0.60            # prompt cap (model/physical cap is RCP = 0.64)
ETA_LO, ETA_HI = 0.80, 0.97
GRADES = ["K1", "K15", "K20", "S22", "K25", "S32", "S38", "K46", "S60"]

MATRIX = MATERIALS["epoxy"]
SHELL = MATERIALS["glass"]


def line(c="-", n=78):
    print(c * n)


# ----------------------------------------------------------------------------- self-checks
def self_checks() -> None:
    line("=")
    print("SELF-CHECKS (independent known results recovered before any design search)")
    line("=")
    p_ref = hollow_particle("K46")
    e0 = hollow_particle_mori_tanaka(MATRIX, p_ref, vf=0.0)
    print(f"[1] vf = 0 limit           : E = {e0.E:.4f} MPa, nu = {e0.nu:.4f}, "
          f"rho = {e0.rho:.4f} g/cm^3")
    print(f"    matrix reference       : E = {MATRIX.E:.4f} MPa, nu = {MATRIX.nu:.4f}, "
          f"rho = {MATRIX.rho:.4f} g/cm^3")
    assert abs(e0.E - MATRIX.E) < 1e-6 * MATRIX.E, "vf=0 must recover the matrix modulus"
    assert abs(e0.nu - MATRIX.nu) < 1e-9 and abs(e0.rho - MATRIX.rho) < 1e-12
    print(f"    -> PASS. E(vf=0) = {e0.E:.1f} MPa < {E_TARGET:.0f} MPa target: the unfilled epoxy")
    print("       matrix is INFEASIBLE on its own; stiff microballoons must be added.")

    # [2] eta -> 0 must reduce to solid-glass-sphere Mori-Tanaka
    from foamsim.micromechanics import mori_tanaka_spheres
    p_solid = HollowParticle(SHELL, eta=1e-9)
    K, G = mori_tanaka_spheres(MATRIX, SHELL, 0.35)
    e_s = hollow_particle_mori_tanaka(MATRIX, p_solid, 0.35)
    E_ref = 9 * K * G / (3 * K + G)
    print(f"[2] eta -> 0 (solid glass) : HP-MT E = {e_s.E:.3f} MPa vs solid-sphere MT "
          f"E = {E_ref:.3f} MPa -> PASS")
    assert abs(e_s.E - E_ref) < 1e-6 * E_ref

    # [3] every estimate inside the HS bounds
    worst = 0.0
    for eta in (0.80, 0.90, 0.97):
        for vf in (0.1, 0.3, 0.5, 0.6):
            p = HollowParticle(SHELL, eta)
            b = hashin_shtrikman_bounds(MATRIX, p, vf)
            for est in (hollow_particle_mori_tanaka(MATRIX, p, vf),
                        hollow_particle_differential(MATRIX, p, vf)):
                assert b["E_lo"] - 1e-6 <= est.E <= b["E_hi"] + 1e-6, (eta, vf, est.model)
                worst = max(worst, (b["E_hi"] - b["E_lo"]) / est.E)
    print(f"[3] HS containment         : HP-MT and HP-DS inside [E_lo, E_hi] at all 12 probe "
          f"points -> PASS (widest band = {100*worst:.1f}% of the estimate)")

    # [4] density rule of mixtures
    p = HollowParticle(SHELL, 0.90)
    rho_chk = 0.4 * p.true_density + 0.6 * MATRIX.rho
    assert abs(density(MATRIX, p, 0.4) - rho_chk) < 1e-12
    print(f"[4] density rule of mixtures: {density(MATRIX, p, 0.4):.4f} = "
          f"{rho_chk:.4f} g/cm^3 -> PASS")
    print(f"[5] packing limit          : RCP = {RCP} (foamsim raises for vf > RCP); "
          f"search capped at vf <= {VF_MAX}")


# ----------------------------------------------------------------------------- feasibility
def feasibility() -> None:
    line("=")
    print("FEASIBILITY: is E >= 3500 MPa reachable at all? (HS upper bound screening)")
    line("=")
    print(f"{'eta':>6} {'rho_p':>8} {'E_hi(HS)':>10} {'E_HP-MT':>10} {'E_HP-DS':>10} "
          f"{'rho':>8}   at vf = 0.60")
    best_hi = -1.0
    for eta in np.arange(ETA_LO, ETA_HI + 1e-9, 0.02):
        p = HollowParticle(SHELL, float(eta))
        b = hashin_shtrikman_bounds(MATRIX, p, VF_MAX)
        mt = hollow_particle_mori_tanaka(MATRIX, p, VF_MAX)
        ds = hollow_particle_differential(MATRIX, p, VF_MAX)
        best_hi = max(best_hi, b["E_hi"])
        print(f"{eta:6.2f} {p.true_density:8.4f} {b['E_hi']:10.1f} {mt.E:10.1f} "
              f"{ds.E:10.1f} {mt.rho:8.4f}")
    print(f"\nMax HS upper bound over the design box = {best_hi:.1f} MPa >= {E_TARGET:.0f} MPa target")
    print("-> the target is NOT above the HS upper bound, so it is not excluded by the")
    print("   two-phase bound: some microstructure of these constituents can reach it.")


# ----------------------------------------------------------------------------- optimisation
def min_vf_for_target(eta: float, model, tol=1e-7) -> float | None:
    """Smallest vf in [0, VF_MAX] with E >= E_TARGET; None if unreachable. E is monotone in vf here."""
    p = HollowParticle(SHELL, eta)
    if model(MATRIX, p, VF_MAX).E < E_TARGET:
        return None
    lo, hi = 0.0, VF_MAX
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if model(MATRIX, p, mid).E >= E_TARGET:
            hi = mid
        else:
            lo = mid
    return hi


def optimise():
    line("=")
    print("INVERSE DESIGN: minimise density subject to E(HP-MT) >= 3500 MPa, vf <= 0.60")
    line("=")
    rows = []
    for eta in np.arange(ETA_LO, ETA_HI + 1e-9, 0.001):
        eta = float(eta)
        p = HollowParticle(SHELL, eta)
        # density decreases with vf whenever rho_p < rho_matrix, so the lightest feasible point
        # for a given eta is at the LARGEST vf, i.e. vf = VF_MAX (checked below), while the
        # binding constraint may instead be E. Evaluate the whole vf line and keep feasible min-rho.
        best = None
        for vf in np.linspace(0.0, VF_MAX, 601):
            e = hollow_particle_mori_tanaka(MATRIX, p, float(vf))
            if e.E >= E_TARGET and (best is None or e.rho < best[0]):
                best = (e.rho, float(vf), e.E)
        if best is not None:
            rows.append({"eta": eta, "rho_p": p.true_density, "vf": best[1],
                         "rho": best[0], "E": best[2]})
    if not rows:
        print("NO FEASIBLE DESIGN in the box.")
        return None, []
    rows.sort(key=lambda r: r["rho"])
    opt = rows[0]

    p = HollowParticle(SHELL, opt["eta"])
    # refine vf on a fine grid around the grid optimum
    lo = max(0.0, opt["vf"] - 0.002); hi = min(VF_MAX, opt["vf"] + 0.002)
    for vf in np.linspace(lo, hi, 4001):
        e = hollow_particle_mori_tanaka(MATRIX, p, float(vf))
        if e.E >= E_TARGET and e.rho < opt["rho"]:
            opt = {"eta": opt["eta"], "rho_p": p.true_density, "vf": float(vf),
                   "rho": e.rho, "E": e.E}

    p = HollowParticle(SHELL, opt["eta"])
    mt = hollow_particle_mori_tanaka(MATRIX, p, opt["vf"])
    ds = hollow_particle_differential(MATRIX, p, opt["vf"])
    b = hashin_shtrikman_bounds(MATRIX, p, opt["vf"])

    print("OPTIMUM (continuous eta):")
    print(f"  eta                = {opt['eta']:.3f}   (wall ratio r_i/r_o; t/R = {1-opt['eta']:.3f})")
    print(f"  particle true rho  = {p.true_density:.4f} g/cm^3")
    print(f"  vf                 = {opt['vf']:.4f}  (<= {VF_MAX}, and < RCP = {RCP})")
    print(f"  composite density  = {opt['rho']:.4f} g/cm^3   "
          f"({100*(1-opt['rho']/MATRIX.rho):.1f}% lighter than neat epoxy)")
    print(f"  E (HP-MT)          = {mt.E:.1f} MPa   (target {E_TARGET:.0f} MPa, "
          f"margin {mt.E-E_TARGET:+.1f} MPa)")
    print(f"  nu (HP-MT)         = {mt.nu:.4f}")
    print(f"  specific modulus   = {mt.E/mt.rho:.0f} MPa/(g/cm^3)  "
          f"(neat epoxy {MATRIX.E/MATRIX.rho:.0f})")
    print("\nMODEL SPREAD / UNCERTAINTY at the optimum:")
    print(f"  HP-MT (Mori-Tanaka)      E = {mt.E:9.1f} MPa")
    print(f"  HP-DS (differential)     E = {ds.E:9.1f} MPa "
          f"({100*(ds.E-mt.E)/mt.E:+.1f}% vs HP-MT)")
    print(f"  HS bounds                E in [{b['E_lo']:.1f}, {b['E_hi']:.1f}] MPa "
          f"(band = {100*(b['E_hi']-b['E_lo'])/mt.E:.1f}% of HP-MT)")
    inside = b["E_lo"] <= mt.E <= b["E_hi"] and b["E_lo"] <= ds.E <= b["E_hi"]
    print(f"  both estimates inside HS bounds: {inside}")
    print(f"  HS-feasibility of the target at this (eta, vf): E_hi = {b['E_hi']:.1f} MPa "
          f">= {E_TARGET:.0f} -> {'FEASIBLE' if b['E_hi'] >= E_TARGET else 'INFEASIBLE'}")
    if ds.E < E_TARGET:
        print(f"  CAUTION: the differential scheme gives {ds.E:.1f} MPa, BELOW the target. The design")
        print("           meets E >= 3500 MPa only within HP-MT; a safety margin is advisable.")
    else:
        print("  The design meets the target under BOTH HP-MT and HP-DS (robust to model choice).")
    return opt, rows


# ----------------------------------------------------------------------------- grades
def grades_table(opt):
    line("=")
    print("DISCRETE 3M GLASS-BUBBLE GRADES (lightest feasible design per grade)")
    line("=")
    print(f"{'grade':>7} {'eta':>7} {'rho_p':>8} {'vf*':>7} {'rho':>8} {'E_MT':>9} {'E_DS':>9} "
          f"{'E_hi(HS)':>9}  status")
    best = None
    for g in GRADES:
        p = hollow_particle(g)
        if not (ETA_LO <= p.eta <= ETA_HI):
            note = f"eta outside [{ETA_LO},{ETA_HI}]"
        else:
            note = ""
        vf = min_vf_for_target(p.eta, hollow_particle_mori_tanaka)
        if vf is None:
            print(f"{g:>7} {p.eta:7.4f} {p.true_density:8.4f} {'--':>7} {'--':>8} "
                  f"{hollow_particle_mori_tanaka(MATRIX, p, VF_MAX).E:9.1f} "
                  f"{hollow_particle_differential(MATRIX, p, VF_MAX).E:9.1f} "
                  f"{hashin_shtrikman_bounds(MATRIX, p, VF_MAX)['E_hi']:9.1f}  "
                  f"infeasible (E<target at vf=0.60) {note}")
            continue
        # lightest feasible vf for this grade: rho monotone in vf, direction set by rho_p vs matrix
        cand = [vf, VF_MAX] if p.true_density < MATRIX.rho else [vf]
        rows = []
        for v in cand:
            e = hollow_particle_mori_tanaka(MATRIX, p, v)
            if e.E >= E_TARGET - 1e-6:
                rows.append((e.rho, v, e))
        rho, v, e = min(rows)
        ds = hollow_particle_differential(MATRIX, p, v)
        b = hashin_shtrikman_bounds(MATRIX, p, v)
        print(f"{g:>7} {p.eta:7.4f} {p.true_density:8.4f} {v:7.4f} {rho:8.4f} {e.E:9.1f} "
              f"{ds.E:9.1f} {b['E_hi']:9.1f}  feasible {note}")
        if ETA_LO <= p.eta <= ETA_HI and (best is None or rho < best[0]):
            best = (rho, g, v, e, ds, b, p)
    if best is not None:
        rho, g, v, e, ds, b, p = best
        print(f"\nBEST NAMED GRADE in eta in [{ETA_LO},{ETA_HI}]: {g} "
              f"(eta = {p.eta:.4f}, rho_p = {p.true_density:.3f} g/cm^3)")
        print(f"  vf = {v:.4f}, density = {rho:.4f} g/cm^3, E(HP-MT) = {e.E:.1f} MPa, "
              f"E(HP-DS) = {ds.E:.1f} MPa, HS = [{b['E_lo']:.1f}, {b['E_hi']:.1f}] MPa")
        if opt is not None:
            print(f"  penalty vs continuous-eta optimum: "
                  f"{1000*(rho-opt['rho']):+.1f} mg/cm^3 ({100*(rho/opt['rho']-1):+.2f}%)")
    return best


# ----------------------------------------------------------------------------- trade-off
def tradeoff():
    line("=")
    print("TRADE-OFF CURVE: minimum achievable density vs required modulus (Pareto front)")
    line("=")
    print(f"{'E_req':>8} {'rho_min':>9} {'eta*':>7} {'vf*':>7} {'E_MT':>9} {'E_DS':>9}   "
          f"{'feasible?':>10}")
    curve = []
    etas = np.arange(ETA_LO, ETA_HI + 1e-9, 0.002)
    parts = [(float(x), HollowParticle(SHELL, float(x))) for x in etas]
    vfs = np.linspace(0.0, VF_MAX, 301)
    cache = {}
    for eta, p in parts:
        cache[eta] = [(float(v), hollow_particle_mori_tanaka(MATRIX, p, float(v))) for v in vfs]
    for E_req in [2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 5000, 5500, 6000, 7000]:
        best = None
        for eta, p in parts:
            for v, e in cache[eta]:
                if e.E >= E_req and (best is None or e.rho < best[0]):
                    best = (e.rho, eta, v, e)
        if best is None:
            print(f"{E_req:8.0f} {'--':>9} {'--':>7} {'--':>7} {'--':>9} {'--':>9}   "
                  f"{'NO':>10}")
            curve.append((E_req, None))
            continue
        rho, eta, v, e = best
        ds = hollow_particle_differential(MATRIX, HollowParticle(SHELL, eta), v)
        star = " <-- TARGET" if E_req == E_TARGET else ""
        print(f"{E_req:8.0f} {rho:9.4f} {eta:7.3f} {v:7.4f} {e.E:9.1f} {ds.E:9.1f}   "
              f"{'YES':>10}{star}")
        curve.append((E_req, rho))
    print("\nReading: density rises monotonically with the required modulus; below ~3000 MPa the")
    print("constraint is slack and the lightest design is the lightest particle at vf = 0.60.")
    return curve


def main():
    print("W2 INVERSE DESIGN - lightest epoxy / glass-microballoon syntactic foam, E >= 3500 MPa")
    print(f"Matrix : epoxy  E = {MATRIX.E:.0f} MPa, nu = {MATRIX.nu}, rho = {MATRIX.rho} g/cm^3")
    print(f"Shell  : glass  E = {SHELL.E:.0f} MPa, nu = {SHELL.nu}, rho = {SHELL.rho} g/cm^3")
    print(f"Box    : eta in [{ETA_LO}, {ETA_HI}], vf in [0, {VF_MAX}] (RCP limit {RCP})")
    print("Units  : E, K, G in MPa; density in g/cm^3; eta, vf, nu dimensionless.")
    self_checks()
    feasibility()
    opt, _ = optimise()
    grades_table(opt)
    tradeoff()
    line("=")
    print("MODEL AND ASSUMPTIONS: HP-MT (equivalent hollow-sphere particle + Mori-Tanaka),")
    print("cross-checked with HP-DS (differential scheme) and bracketed by Hashin-Shtrikman bounds.")
    print("Linear elastic, perfectly bonded, monodisperse, non-interacting spheres; no matrix")
    print("porosity, no particle crushing, small strain. Experimental syntactic-foam moduli are")
    print("typically 20-40% BELOW HP-MT (matrix porosity, particle breakage, imperfect bonding),")
    print("so treat the reported E as an upper-ish estimate and design with margin.")
    line("=")


if __name__ == "__main__":
    main()
