"""Modulus + density of epoxy / K46 syntactic foam.

The requested operating point (vf = 0.75 monodisperse, eta = 1.02) is not physically
realisable, so this script (1) documents both violations explicitly and (2) reports the
result at the closest realisable conditions instead.

Why the request is ill-posed
  * vf = 0.75 exceeds random close packing of monodisperse spheres (RCP = 0.64).
    75 vol% is only reachable with a polydisperse / graded size distribution, which the
    prompt explicitly excludes ("monodisperse").
  * eta = r_inner / r_outer must lie in [0, 1). eta = 1.02 would mean the inner radius
    exceeds the outer radius (negative wall thickness). K46's real wall ratio, inferred
    from its 0.46 g/cm^3 true density, is ~0.937.
"""

from foamsim import MATERIALS, hollow_particle
from foamsim.materials import HollowParticle
from foamsim.micromechanics import (
    RCP,
    density,
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
)

VF_REQ, ETA_REQ = 0.75, 1.02

matrix = MATERIALS["epoxy"]
p_k46 = hollow_particle("K46")

print("=== Requested point: vf=0.75 (monodisperse), eta=1.02 ===")

# 1. eta = 1.02 -- invalid by construction.
try:
    HollowParticle(MATERIALS["glass"], eta=ETA_REQ, diameter_um=40.0)
    print("eta=1.02 accepted (unexpected)")
except ValueError as exc:
    print(f"eta={ETA_REQ} REJECTED: {exc}")
    print("  -> wall ratio >= 1 means a negative wall thickness; no such microballoon exists.")

# 2. vf = 0.75 -- above random close packing.
try:
    hollow_particle_mori_tanaka(matrix, p_k46, vf=VF_REQ)
    print("vf=0.75 accepted (unexpected)")
except ValueError as exc:
    print(f"vf={VF_REQ} REJECTED: {exc}")
    print(f"  -> monodisperse spheres cannot exceed RCP = {RCP}.")

print(f"\nK46 as supplied: eta = {p_k46.eta:.4f} (true density {p_k46.true_density:.3f} g/cm^3), "
      f"diameter {p_k46.diameter_um:.0f} um")

# 3. Closest realisable point: real K46 wall ratio, vf at the monodisperse packing limit.
vf = RCP
mt = hollow_particle_mori_tanaka(matrix, p_k46, vf=vf)
ds = hollow_particle_differential(matrix, p_k46, vf=vf)
hs = hashin_shtrikman_bounds(matrix, p_k46, vf=vf)
rho = density(matrix, p_k46, vf=vf)

print(f"\n=== Closest realisable point: K46 (eta={p_k46.eta:.4f}), vf={vf} (= RCP) ===")
print(f"Matrix: epoxy E={matrix.E} MPa, nu={matrix.nu}, rho={matrix.rho} g/cm^3")
print(f"HP-Mori-Tanaka : E = {mt.E:.1f} MPa, nu = {mt.nu:.4f}")
print(f"HP-Differential: E = {ds.E:.1f} MPa, nu = {ds.nu:.4f}")
print(f"HS bounds      : E_lo = {hs['E_lo']:.1f} MPa, E_hi = {hs['E_hi']:.1f} MPa")
print(f"Density        : rho = {rho:.4f} g/cm^3")

for name, est in (("MT", mt), ("DS", ds)):
    inside = hs["E_lo"] <= est.E <= hs["E_hi"]
    print(f"  self-check: {name} inside HS band -> {inside}")

# Sanity limit: vf = 0 must return the matrix.
e0 = hollow_particle_mori_tanaka(matrix, p_k46, vf=0.0)
print(f"  self-check: vf=0 -> E = {e0.E:.1f} MPa (matrix {matrix.E} MPa)")

print("\nCAVEAT: the numbers above are NOT at the requested vf=0.75 / eta=1.02, which are "
      "unphysical. They are at K46's actual wall ratio and the monodisperse packing limit "
      "vf=0.64. To genuinely reach 75 vol% a polydisperse microballoon blend is required, "
      "and this analytical model is not calibrated for it.")
