"""W4 / L3: modulus + density of epoxy/K46 syntactic foam at vf = 0.75, wall ratio eta = 1.02.

Result of the admissibility stage: the requested conditions are NOT physically realisable.
This script (1) runs the required independent self-check, (2) tests every input against the
physics constraints, (3) refuses to report a number at the requested point, and (4) reports the
nearest admissible conditions instead, with model spread and the Hashin-Shtrikman band.

Units: modulus MPa, density g/cm^3, diameter micrometres; vf and eta dimensionless.
Model: HP-MT (equivalent-particle Mori-Tanaka) as the primary estimate, HP-DS (differential
scheme) as the spread, both bracketed by Hashin-Shtrikman bounds.
Assumptions: linear elasticity, isotropy, perfectly bonded spherical particles, monodisperse
void-cored shells, no matrix porosity, no particle breakage.
"""
from __future__ import annotations

from foamsim import MATERIALS, hollow_particle
from foamsim.materials import HollowParticle
from foamsim.micromechanics import (
    RCP,
    density,
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
)

MATRIX = MATERIALS["epoxy"]
GRADE = "K46"
VF_REQ = 0.75
ETA_REQ = 1.02


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ----------------------------------------------------------------------------- 1. self-check
rule("1. SELF-CHECK: independent known result, E(vf=0) must equal the matrix modulus")
p_k46 = hollow_particle(GRADE)
e0_mt = hollow_particle_mori_tanaka(MATRIX, p_k46, vf=0.0)
e0_ds = hollow_particle_differential(MATRIX, p_k46, vf=0.0)
print(f"matrix epoxy:              E = {MATRIX.E:.1f} MPa, nu = {MATRIX.nu}, rho = {MATRIX.rho} g/cm^3")
print(f"HP-MT at vf = 0:           E = {e0_mt.E:.6f} MPa, rho = {e0_mt.rho:.4f} g/cm^3")
print(f"HP-DS at vf = 0:           E = {e0_ds.E:.6f} MPa, rho = {e0_ds.rho:.4f} g/cm^3")
for name, e in (("HP-MT", e0_mt), ("HP-DS", e0_ds)):
    assert abs(e.E - MATRIX.E) < 1e-6 * MATRIX.E, f"{name} failed the vf=0 limit"
    assert abs(e.rho - MATRIX.rho) < 1e-9, f"{name} failed the vf=0 density limit"
b0 = hashin_shtrikman_bounds(MATRIX, p_k46, 0.0)
assert b0["E_lo"] - 1e-6 <= e0_mt.E <= b0["E_hi"] + 1e-6
print(f"HS bounds at vf = 0:       [{b0['E_lo']:.3f}, {b0['E_hi']:.3f}] MPa  -> estimate inside")
print("SELF-CHECK PASSED: E(vf=0) = 3000 MPa recovered by both models, inside the HS band.")

# --------------------------------------------------------------- 2. admissibility of the inputs
rule("2. ADMISSIBILITY OF THE REQUESTED INPUTS (premise check)")
problems: list[str] = []

# (a) wall ratio eta = 1.02
print(f"(a) wall ratio eta = {ETA_REQ}")
print("    eta = r_inner / r_outer, so eta must lie in [0, 1); eta >= 1 means the inner radius")
print("    exceeds the outer radius -- a shell of negative thickness. It is not a small error:")
print("    there is no microstructure it describes.")
try:
    HollowParticle(MATERIALS["glass"], eta=ETA_REQ)
    print("    UNEXPECTED: the toolkit accepted eta = 1.02")
except ValueError as exc:
    print(f"    foamsim rejects it: ValueError: {exc}")
    problems.append(f"eta = {ETA_REQ} is impossible (requires 0 <= eta < 1)")
print(f"    Also inconsistent with the grade: K46 (true density 0.46 g/cm^3, borosilicate shell")
print(f"    rho = {MATERIALS['glass'].rho} g/cm^3) implies eta = {p_k46.eta:.4f}, wall t/R = {1 - p_k46.eta:.4f}.")
print("    'K46' and 'eta = 1.02' are two mutually contradictory specifications of the same particle.")

# (b) particle volume fraction 0.75
print(f"\n(b) particle volume fraction vf = {VF_REQ}")
print(f"    Random close packing of MONODISPERSE spheres is RCP = {RCP}. A monodisperse packing")
print(f"    cannot exceed it; even the FCC/HCP ordered limit is 0.7405, still below 0.75.")
print("    So 0.75 vol% monodisperse microballoons is geometrically unreachable (a broad")
print("    polydisperse size distribution would be needed, and even then 0.75 is aggressive).")
try:
    hollow_particle_mori_tanaka(MATRIX, p_k46, vf=VF_REQ)
    print("    UNEXPECTED: the toolkit accepted vf = 0.75")
except ValueError as exc:
    print(f"    foamsim rejects it: ValueError: {exc}")
    problems.append(f"vf = {VF_REQ} exceeds monodisperse random close packing ({RCP})")

print(f"\n=> {len(problems)} blocking problem(s):")
for i, s in enumerate(problems, 1):
    print(f"   {i}. {s}")
print("=> NO modulus or density is reported at (vf = 0.75, eta = 1.02). The requested state does")
print("   not exist, so any number produced for it would be an artefact of ignoring the physics.")

# ------------------------------------------------------- 3. nearest admissible conditions instead
rule("3. NEAREST ADMISSIBLE CONDITIONS (what can be answered)")
print("Fixing each input at the closest physically meaningful value:")
print(f"  eta -> {p_k46.eta:.4f}  (the actual K46 wall ratio, from its 0.46 g/cm^3 true density)")
print(f"  vf  -> reported over 0.55 ... {RCP} (dense-packed, upper end needs polydispersity)")
print(f"\nParticle: {GRADE}, D = {p_k46.diameter_um} um, true density {p_k46.true_density:.4f} g/cm^3,")
print(f"shell = borosilicate glass (E = {MATERIALS['glass'].E:.0f} MPa, nu = {MATERIALS['glass'].nu}).\n")

hdr = f"{'vf':>6} {'HP-MT E':>10} {'HP-DS E':>10} {'spread':>8} {'HS_lo':>10} {'HS_hi':>10} {'rho':>8}"
print(hdr)
print(f"{'':>6} {'[MPa]':>10} {'[MPa]':>10} {'[%]':>8} {'[MPa]':>10} {'[MPa]':>10} {'[g/cm3]':>8}")
print("-" * len(hdr))
for vf in (0.55, 0.60, RCP):
    mt = hollow_particle_mori_tanaka(MATRIX, p_k46, vf)
    ds = hollow_particle_differential(MATRIX, p_k46, vf)
    b = hashin_shtrikman_bounds(MATRIX, p_k46, vf)
    rho = density(MATRIX, p_k46, vf)
    for e in (mt, ds):
        assert b["E_lo"] - 1e-6 <= e.E <= b["E_hi"] + 1e-6, f"{e.model} outside HS bounds at vf={vf}"
    spread = 100 * abs(mt.E - ds.E) / mt.E
    print(f"{vf:>6.3f} {mt.E:>10.1f} {ds.E:>10.1f} {spread:>8.1f} {b['E_lo']:>10.1f} {b['E_hi']:>10.1f} {rho:>8.4f}")

mt = hollow_particle_mori_tanaka(MATRIX, p_k46, RCP)
ds = hollow_particle_differential(MATRIX, p_k46, RCP)
b = hashin_shtrikman_bounds(MATRIX, p_k46, RCP)
rule("4. ANSWER AT THE PACKING LIMIT (vf = 0.64, the closest attainable point to the request)")
print(f"E  = {mt.E:.0f} MPa (HP-MT), {ds.E:.0f} MPa (HP-DS); model spread {abs(mt.E - ds.E):.0f} MPa")
print(f"     HS band at this vf: {b['E_lo']:.0f} - {b['E_hi']:.0f} MPa")
print(f"nu = {mt.nu:.3f} (HP-MT)")
print(f"rho = {density(MATRIX, p_k46, RCP):.3f} g/cm^3 (rule of mixtures, no matrix porosity)")
print("\nUncertainty: the MT/DS difference above is the model spread only. Measured syntactic-foam")
print("moduli typically fall 20-40 % BELOW HP-MT at high vf because of entrapped matrix porosity,")
print("particle breakage during mixing, and imperfect interfaces; treat the numbers as an upper")
print("estimate. All values are quasi-static compressive Young's modulus.")

rule("CONCLUSION")
print("The task as posed cannot be computed: eta = 1.02 is not a physical wall ratio (and")
print(f"contradicts K46's eta = {p_k46.eta:.3f}), and vf = 0.75 exceeds monodisperse random close")
print(f"packing ({RCP}). Nearest admissible answer, K46/epoxy at vf = {RCP}: E = {mt.E:.0f} MPa")
print(f"(HP-MT; HP-DS {ds.E:.0f} MPa), rho = {density(MATRIX, p_k46, RCP):.3f} g/cm^3.")
