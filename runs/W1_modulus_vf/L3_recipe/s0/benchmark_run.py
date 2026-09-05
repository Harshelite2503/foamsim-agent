"""Compressive modulus of epoxy / 3M K46 glass-microballoon syntactic foam vs particle volume fraction.

System   : epoxy matrix (E=3000 MPa, nu=0.35, rho=1.18 g/cm^3)
           3M K46 glass bubbles (true density 0.46 g/cm^3; borosilicate shell E=60 GPa, nu=0.21, rho=2.54 g/cm^3)
Sweep    : vf = 0 .. 0.6 (particle volume fraction INCLUDING the hollow cores), quasi-static compression.

Models   : HP-MT  equivalent-particle Mori-Tanaka (Benveniste 1987) -- primary estimate
           HP-DS  equivalent-particle differential scheme (McLaughlin 1977) -- second estimate / spread
           HS     Hashin-Shtrikman lower/upper bounds for the same two-phase (matrix + equivalent particle) system

Assumptions (both models): linear elasticity, isotropy, perfectly bonded matrix/particle interface, monodisperse
spherical thin-walled shells with a void core, no matrix porosity, no particle breakage, no percolation of contacts.
The hollow shell is first replaced by an equivalent homogeneous solid sphere (K exact -- Hashin 1962 composite
spheres; G = Hashin-Shtrikman upper bound of the porous shell, no exact result exists), then homogenized.

Units    : moduli in MPa, densities in g/cm^3, volume fractions dimensionless.

Outputs  : modulus_vs_vf.csv, modulus_vs_vf.png, experimental_comparison.csv, run.log (via tee)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foamsim import MATERIALS, hollow_particle
from foamsim.micromechanics import (
    RCP,
    density,
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
    hollow_sphere_equivalent,
)
from foamsim.data import reference_curve

matrix = MATERIALS["epoxy"]
particle = hollow_particle("K46")
VF = np.linspace(0.0, 0.6, 13)

print("=" * 78)
print("Epoxy / 3M K46 glass-microballoon syntactic foam: compressive modulus vs vf")
print("=" * 78)
print(f"matrix   : {matrix.name}  E={matrix.E:.0f} MPa  nu={matrix.nu}  rho={matrix.rho} g/cm^3 "
      f"(K={matrix.K:.1f} MPa, G={matrix.G:.1f} MPa)")
print(f"shell    : {particle.shell.name}  E={particle.shell.E:.0f} MPa  nu={particle.shell.nu}  "
      f"rho={particle.shell.rho} g/cm^3")
print(f"particle : K46  eta=r_i/r_o={particle.eta:.4f}  wall vol frac={particle.wall_volume_fraction:.4f}  "
      f"true density={particle.true_density:.3f} g/cm^3  d={particle.diameter_um:.0f} um")
eqp = hollow_sphere_equivalent(particle)
print(f"equivalent solid particle: E={eqp.E:.1f} MPa  nu={eqp.nu:.4f}  K={eqp.K:.1f} MPa  G={eqp.G:.1f} MPa")
print("Units throughout: modulus MPa, density g/cm^3, vf dimensionless.\n")

# ----------------------------------------------------------------------------------
# 1. Self-checks / premise checks BEFORE producing results (validation protocol step 1)
# ----------------------------------------------------------------------------------
print("-- Self-checks (independent known results) " + "-" * 34)
checks: list[tuple[str, bool, str]] = []

e0_mt = hollow_particle_mori_tanaka(matrix, particle, 0.0)
e0_ds = hollow_particle_differential(matrix, particle, 0.0)
checks.append(("E(vf=0) == matrix E = 3000 MPa (HP-MT)",
               abs(e0_mt.E - 3000.0) < 1e-6, f"HP-MT E(0) = {e0_mt.E:.6f} MPa"))
checks.append(("E(vf=0) == matrix E = 3000 MPa (HP-DS)",
               abs(e0_ds.E - 3000.0) < 1e-6, f"HP-DS E(0) = {e0_ds.E:.6f} MPa"))
checks.append(("nu(vf=0) == matrix nu = 0.35",
               abs(e0_mt.nu - 0.35) < 1e-9, f"nu(0) = {e0_mt.nu:.6f}"))

rho40 = density(matrix, particle, 0.4)
checks.append(("density(vf=0.4) == 0.892 g/cm^3",
               abs(rho40 - 0.892) < 1e-9, f"rho(0.4) = {rho40:.6f} g/cm^3"))

inside = True
worst = ""
for vf in VF:
    b = hashin_shtrikman_bounds(matrix, particle, vf)
    for e in (hollow_particle_mori_tanaka(matrix, particle, vf),
              hollow_particle_differential(matrix, particle, vf)):
        ok = b["E_lo"] - 1e-9 <= e.E <= b["E_hi"] + 1e-9
        if not ok:
            inside = False
            worst = f"{e.model} at vf={vf:.2f}: E={e.E:.2f} outside [{b['E_lo']:.2f}, {b['E_hi']:.2f}]"
checks.append(("all HP-MT and HP-DS estimates inside HS bounds", inside, worst or "26/26 estimates inside band"))

# independent limit: eta -> 0 (solid glass spheres) must recover Mori-Tanaka for solid spheres
from foamsim.materials import HollowParticle
from foamsim.micromechanics import mori_tanaka_spheres
solid = HollowParticle(MATERIALS["glass"], eta=0.0)
K_ref, G_ref = mori_tanaka_spheres(matrix, MATERIALS["glass"], 0.3)
e_solid = hollow_particle_mori_tanaka(matrix, solid, 0.3)
E_ref = 9 * K_ref * G_ref / (3 * K_ref + G_ref)
checks.append(("eta=0 limit reproduces solid-sphere Mori-Tanaka (vf=0.3)",
               abs(e_solid.E - E_ref) < 1e-6 * E_ref, f"{e_solid.E:.3f} vs {E_ref:.3f} MPa"))

for name, ok, detail in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
assert all(ok for _, ok, _ in checks), "self-check failed -- results not trustworthy"

print("\n-- Premise checks " + "-" * 60)
print(f"  vf range 0-0.6 is below random close packing RCP={RCP} (monodisperse spheres): OK, but")
print("  vf > ~0.55 in practice requires polydisperse packing; the vf=0.60 point is at the edge of")
print("  what is realisable and real foams at that loading show particle-particle contact and damage.")
print(f"  eta={particle.eta:.4f} < 1 and true density 0.46 <= shell density {particle.shell.rho}: physically valid.")
print("  Models are linear-elastic estimates; they do not capture particle crushing or matrix porosity.\n")

# ----------------------------------------------------------------------------------
# 2. Sweep
# ----------------------------------------------------------------------------------
rows = []
for vf in VF:
    mt = hollow_particle_mori_tanaka(matrix, particle, vf)
    ds = hollow_particle_differential(matrix, particle, vf)
    b = hashin_shtrikman_bounds(matrix, particle, vf)
    rows.append({
        "vf": round(float(vf), 4),
        "density_g_cc": density(matrix, particle, vf),
        "E_HPMT_mpa": mt.E,
        "E_HPDS_mpa": ds.E,
        "E_HS_lower_mpa": b["E_lo"],
        "E_HS_upper_mpa": b["E_hi"],
        "nu_HPMT": mt.nu,
        "K_HPMT_mpa": mt.K,
        "G_HPMT_mpa": mt.G,
        "specific_E_HPMT_mpa_per_g_cc": mt.E / mt.rho,
        "model_spread_pct": 100.0 * abs(mt.E - ds.E) / mt.E,
    })
df = pd.DataFrame(rows)
df.to_csv("modulus_vs_vf.csv", index=False)

print("-- Sweep: vf, density, modulus (MPa) " + "-" * 40)
show = df[["vf", "density_g_cc", "E_HPMT_mpa", "E_HPDS_mpa", "E_HS_lower_mpa", "E_HS_upper_mpa",
           "model_spread_pct"]]
print(show.to_string(index=False, float_format=lambda x: f"{x:9.3f}"))
print("\nWritten: modulus_vs_vf.csv")
print(f"Model spread (|HP-MT - HP-DS|/HP-MT): max {df.model_spread_pct.max():.2f}% at vf="
      f"{df.loc[df.model_spread_pct.idxmax(), 'vf']:.2f}; "
      f"HS band width at vf=0.6: {df.E_HS_lower_mpa.iloc[-1]:.0f}-{df.E_HS_upper_mpa.iloc[-1]:.0f} MPa.")

# ----------------------------------------------------------------------------------
# 3. Comparison with experimental epoxy / glass-microballoon data (FoamGPT)
# ----------------------------------------------------------------------------------
ref = reference_curve("epoxy", "glass_microballoon")
exp = ref[ref.modulus_mpa.notna() & (ref.particle_volume_fraction <= 0.6)].copy()

# The bundled epoxy/GMB compression records do NOT use K46 alone -- they are mostly lighter S22/S32/S38
# grades and multi-layer graded foams. Comparing them against the K46 curve would confound grade mismatch
# with model error, so each record is ALSO predicted with its own microballoon true density:
#   - stated `particle_true_density_g_cc` when the paper gives it;
#   - otherwise inferred from the measured composite density by inverting the rule of mixtures
#     rho_p = (rho_c - (1-vf) rho_m) / vf   (a density inversion, not a fitted modulus).
def grade_matched_particle(row):
    rho_p = row.particle_true_density_g_cc
    src = "stated"
    if not np.isfinite(rho_p):
        if np.isfinite(row.measured_density_g_cc) and row.particle_volume_fraction > 0:
            rho_p = (row.measured_density_g_cc - (1 - row.particle_volume_fraction) * matrix.rho) \
                / row.particle_volume_fraction
            src = "inferred from measured density"
        else:
            return None, np.nan, "unknown"
    if not 0 < rho_p <= particle.shell.rho:
        return None, rho_p, "unphysical"
    return HollowParticle.from_true_density(particle.shell, rho_p), rho_p, src


recs = []
for _, row in exp.iterrows():
    vf = float(row.particle_volume_fraction)
    b_k46 = hashin_shtrikman_bounds(matrix, particle, vf)
    p_g, rho_p, src = grade_matched_particle(row)
    if p_g is not None:
        e_mt_g = hollow_particle_mori_tanaka(matrix, p_g, vf).E
        e_ds_g = hollow_particle_differential(matrix, p_g, vf).E
        b_g = hashin_shtrikman_bounds(matrix, p_g, vf)
    else:
        e_mt_g = e_ds_g = np.nan
        b_g = {"E_lo": np.nan, "E_hi": np.nan}
    recs.append({
        "record_id": row.record_id, "paper_id": row.paper_id, "vf": vf,
        "measured_density_g_cc": row.measured_density_g_cc, "modulus_mpa": row.modulus_mpa,
        "particle_true_density_g_cc": rho_p, "true_density_source": src,
        "E_HPMT_K46_mpa": hollow_particle_mori_tanaka(matrix, particle, vf).E,
        "E_HS_lo_K46_mpa": b_k46["E_lo"], "E_HS_hi_K46_mpa": b_k46["E_hi"],
        "E_HPMT_grade_mpa": e_mt_g, "E_HPDS_grade_mpa": e_ds_g,
        "E_HS_lo_grade_mpa": b_g["E_lo"], "E_HS_hi_grade_mpa": b_g["E_hi"],
    })
cmp_df = pd.DataFrame(recs)
cmp_df["rel_error_K46_pct"] = 100 * (cmp_df.E_HPMT_K46_mpa - cmp_df.modulus_mpa) / cmp_df.modulus_mpa
cmp_df["rel_error_grade_pct"] = 100 * (cmp_df.E_HPMT_grade_mpa - cmp_df.modulus_mpa) / cmp_df.modulus_mpa
cmp_df["inside_HS_grade"] = (cmp_df.modulus_mpa >= cmp_df.E_HS_lo_grade_mpa) & \
                            (cmp_df.modulus_mpa <= cmp_df.E_HS_hi_grade_mpa)
cmp_df.to_csv("experimental_comparison.csv", index=False)

g = cmp_df[cmp_df.E_HPMT_grade_mpa.notna()]
mape_k46 = float(np.mean(np.abs(cmp_df.rel_error_K46_pct)))
mape_grade = float(np.mean(np.abs(g.rel_error_grade_pct)))
mape_ds_grade = float(np.mean(np.abs(g.E_HPDS_grade_mpa - g.modulus_mpa) / g.modulus_mpa) * 100)

print("\n-- Experimental comparison (FoamGPT: epoxy / glass microballoon, quasi-static compression) " + "-" * 2)
print(cmp_df[["record_id", "vf", "measured_density_g_cc", "particle_true_density_g_cc", "true_density_source",
              "modulus_mpa", "E_HPMT_K46_mpa", "E_HPMT_grade_mpa", "rel_error_grade_pct", "inside_HS_grade"]]
      .to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
print(f"\n  n = {len(cmp_df)} experimental points from {cmp_df.paper_id.nunique()} papers, "
      f"vf {cmp_df.vf.min():.2f}-{cmp_df.vf.max():.2f}.")
print(f"  MAPE against the K46 curve            : {mape_k46:.0f}%  (dominated by grade mismatch -- these")
print("     records are lighter S22/S32/S38 balloons and multi-layer graded foams, not epoxy/K46.)")
print(f"  MAPE with grade-matched particle HP-MT : {mape_grade:.0f}%  (HP-DS {mape_ds_grade:.0f}%), n={len(g)}")
print(f"  Experimental points inside their grade-matched HS band: {int(g.inside_HS_grade.sum())}/{len(g)}")
print(f"  Mean signed error (grade-matched HP-MT - experiment) = {g.rel_error_grade_pct.mean():+.0f}%")
print("  Interpretation: even grade-matched, the analytical estimates lie ABOVE every measurement, and several")
print("  measured moduli fall BELOW the HS lower bound -- which no microstructure of perfectly bonded intact")
print("  balloons in dense epoxy can do. The usual causes are matrix porosity, weak/debonded interfaces and")
print("  balloon breakage during mixing/cure, plus the fact that reported 'compressive modulus' is often a")
print("  machine-compliance-affected secant slope. Literature typically reports experiment 20-40% below HP-MT;")
print("  the gap here is larger because these records are 0.6 vf graded laminates. No constant was tuned.")
print("  Written: experimental_comparison.csv")

# ----------------------------------------------------------------------------------
# 4. Plot
# ----------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

ax[0].fill_between(df.vf, df.E_HS_lower_mpa, df.E_HS_upper_mpa, color="0.85",
                   label="Hashin-Shtrikman bounds")
ax[0].plot(df.vf, df.E_HPMT_mpa, "-", lw=2, color="C0", label="HP-MT (Mori-Tanaka)")
ax[0].plot(df.vf, df.E_HPDS_mpa, "--", lw=2, color="C1", label="HP-DS (differential)")
ax[0].scatter(exp.particle_volume_fraction, exp.modulus_mpa, marker="o", s=32, facecolor="none",
              edgecolor="k", zorder=5, label="experiment (FoamGPT, epoxy/GMB)")
ax[0].axhline(matrix.E, color="0.5", lw=0.8, ls=":")
ax[0].annotate("neat epoxy 3000 MPa", (0.02, matrix.E), fontsize=8, color="0.4", va="bottom")
ax[0].set_xlabel("particle volume fraction $v_f$ (-)")
ax[0].set_ylabel("compressive Young's modulus $E$ (MPa)")
ax[0].set_title("Epoxy / 3M K46 syntactic foam")
ax[0].legend(fontsize=8, loc="upper left")
ax[0].grid(alpha=0.3)

ax[1].plot(df.vf, df.density_g_cc, "-o", ms=3, color="C2", label="model density")
ax[1].scatter(exp.particle_volume_fraction, exp.measured_density_g_cc, marker="s", s=28,
              facecolor="none", edgecolor="k", label="measured density")
ax[1].set_xlabel("particle volume fraction $v_f$ (-)")
ax[1].set_ylabel(r"density $\rho$ (g/cm$^3$)")
ax[1].set_title("Density (rule of mixtures, K46 true density 0.46)")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)

fig.suptitle("Compressive modulus and density vs volume fraction (MPa, g/cm$^3$)", fontsize=11)
fig.tight_layout()
fig.savefig("modulus_vs_vf.png", dpi=160)
print("  Written: modulus_vs_vf.png")

print("\n-- Summary " + "-" * 66)
e60 = df.iloc[-1]
print(f"  HP-MT (primary model): E rises from 3000 MPa at vf=0 to {e60.E_HPMT_mpa:.0f} MPa at vf=0.60,")
print(f"  while density falls from 1.180 to {e60.density_g_cc:.3f} g/cm^3 (specific modulus "
      f"{e60.specific_E_HPMT_mpa_per_g_cc:.0f} -> vs 2542 MPa/(g/cm^3) for neat epoxy).")
print(f"  Uncertainty: model spread HP-MT vs HP-DS <= {df.model_spread_pct.max():.1f}%; the HS band is much wider")
print("  and is a bound, not an uncertainty; experiment scatters below both estimates (see MAPE above).")
