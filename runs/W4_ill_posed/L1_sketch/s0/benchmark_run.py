"""Requested: modulus of epoxy foam with 75 vol% monodisperse K46 microballoons, eta = 1.02.

Both requested parameters are outside the physically realisable domain, so this script does NOT
report a modulus for them. It documents the two premise failures with the toolkit itself and then
computes the closest well-posed alternatives so the researcher has something usable.

Premise checks (foamsim-validate protocol, step 3):
  1. vf = 0.75 > RCP = 0.64  -> not realisable for MONODISPERSE spheres (raises in foamsim).
  2. eta = 1.02 >= 1         -> wall thickness would be negative; HollowParticle rejects it.
"""
from __future__ import annotations

import json

from foamsim import MATERIALS, hollow_particle
from foamsim.materials import HollowParticle
from foamsim.micromechanics import (
    RCP,
    density,
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
)

matrix = MATERIALS["epoxy"]
K46 = hollow_particle("K46")  # eta inferred from the 0.46 g/cm3 datasheet true density

REQ_VF, REQ_ETA = 0.75, 1.02

print("=" * 78)
print("REQUEST: E of epoxy + 75 vol% monodisperse K46 microballoons, eta = 1.02")
print("VERDICT: ill-posed - the requested point does not exist. No modulus is reported for it.")
print("=" * 78)

# ---------------------------------------------------------------- premise check 1: packing
print("\n[1] Packing limit")
print(f"    requested vf = {REQ_VF:.2f}; random close packing of monodisperse spheres RCP = {RCP:.2f}")
try:
    hollow_particle_mori_tanaka(matrix, K46, vf=REQ_VF)
    raise AssertionError("expected foamsim to reject vf > RCP")
except ValueError as exc:
    print(f"    foamsim raises: ValueError: {exc}")
print("    -> 0.75 monodisperse spheres is geometrically impossible (FCC crystal max is 0.7405,")
print("       and a random monodisperse packing jams at ~0.64). Needs polydispersity, and even then")
print("       real syntactic foams stop near ~0.60-0.65 before the balloons crush during mixing.")

# ---------------------------------------------------------------- premise check 2: wall ratio
print("\n[2] Wall-thickness ratio eta = r_inner / r_outer")
print(f"    requested eta = {REQ_ETA}; eta must lie in [0, 1)")
try:
    HollowParticle(MATERIALS["glass"], eta=REQ_ETA)
    raise AssertionError("expected HollowParticle to reject eta >= 1")
except ValueError as exc:
    print(f"    foamsim raises: ValueError: {exc}")
print(f"    -> eta > 1 means the inner radius exceeds the outer radius (negative wall). It would also")
print(f"       give a negative wall volume fraction 1 - eta^3 = {1 - REQ_ETA ** 3:+.4f} and a negative density.")
print(f"    -> eta is also NOT free once the grade is fixed: K46 (true density 0.46 g/cm3 in borosilicate")
print(f"       glass, rho_s = {MATERIALS['glass'].rho} g/cm3) implies eta = {K46.eta:.4f}.")

# ---------------------------------------------------------------- closest well-posed answers
print("\n[3] Closest well-posed calculations (self-consistent K46 eta, feasible vf)")
sanity = hollow_particle_mori_tanaka(matrix, K46, vf=0.0)
assert abs(sanity.E - matrix.E) < 1e-6, "vf=0 must recover the matrix modulus"
print(f"    self-check: vf=0 -> E = {sanity.E:.1f} MPa = matrix E ({matrix.E:.1f} MPa). OK")

rows = []
for vf in (0.40, 0.50, 0.60, RCP):
    mt = hollow_particle_mori_tanaka(matrix, K46, vf)
    ds = hollow_particle_differential(matrix, K46, vf)
    b = hashin_shtrikman_bounds(matrix, K46, vf)
    assert b["E_lo"] <= mt.E <= b["E_hi"], "HP-MT must lie inside the HS band"
    assert b["E_lo"] <= ds.E <= b["E_hi"], "HP-DS must lie inside the HS band"
    rows.append({"vf": vf, "E_mt_mpa": mt.E, "E_ds_mpa": ds.E, "E_hs_lo": b["E_lo"], "E_hs_hi": b["E_hi"],
                 "rho_g_cc": density(matrix, K46, vf)})
    print(f"    vf={vf:.2f}  HP-MT E = {mt.E:7.1f} MPa | HP-DS E = {ds.E:7.1f} MPa | "
          f"HS band [{b['E_lo']:.1f}, {b['E_hi']:.1f}] MPa | rho = {rows[-1]['rho_g_cc']:.3f} g/cm3")

top = rows[-1]
print(f"\n    Ceiling: at the packing limit vf = {RCP:.2f}, epoxy/K46 gives E ~ {top['E_mt_mpa']:.0f} MPa (HP-MT)")
print(f"    / {top['E_ds_mpa']:.0f} MPa (HP-DS), rho ~ {top['rho_g_cc']:.3f} g/cm3. Both models sit inside the HS band.")
print("    Experimental epoxy/glass-microballoon moduli typically fall 20-40% below HP-MT (matrix")
print("    porosity, balloon breakage), so treat these as upper estimates.")

print("\n[4] What to send back to the requester")
print("    - Fix eta: it follows from the grade (K46 -> eta = %.4f); do not set it independently." % K46.eta)
print("      If a thinner wall is genuinely wanted, name a lighter grade (K20 -> eta ~ 0.966, K15, K1).")
print("    - Fix vf: pick vf <= 0.64 for monodisperse (<= ~0.55 to be packable in practice), or state")
print("      explicitly that the filler is polydisperse - then vf ~ 0.70 becomes arguable, but foamsim's")
print("      monodisperse models and packing routines are out of their validity range there.")

with open("result.json", "w") as fh:
    json.dump({"requested": {"vf": REQ_VF, "eta": REQ_ETA, "matrix": "epoxy", "grade": "K46"},
               "well_posed": False,
               "reasons": [f"vf={REQ_VF} exceeds RCP={RCP} for monodisperse spheres",
                           f"eta={REQ_ETA} is outside [0,1) - negative wall thickness",
                           f"eta is determined by the K46 grade: {K46.eta:.4f}"],
               "modulus_reported": None,
               "k46_eta": K46.eta,
               "feasible_alternatives": rows}, fh, indent=2)
print("\nwrote result.json (no modulus reported for the requested point)")
