"""W1: compressive modulus of epoxy / 3M K46 glass-microballoon syntactic foam vs volume fraction.

System: epoxy matrix (E=3000 MPa, nu=0.35, rho=1.18 g/cm3); 3M K46 glass microballoons
(true density 0.46 g/cm3, borosilicate shell E=60 GPa, nu=0.21, rho=2.54 g/cm3).
Particle volume fraction 0 -> 0.6, quasi-static compression.

Outputs (in this directory):
  modulus_vs_vf.csv   vf, density, HP-MT modulus, HP-DS modulus, HS bounds
  modulus_vs_vf.png   modulus and density vs vf with HS band and experimental points
  experimental_comparison.csv  FoamGPT epoxy/glass-microballoon rows vs model at their vf
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foamsim import MATERIALS, hollow_particle
from foamsim.materials import HollowParticle
from foamsim.micromechanics import (
    density,
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
)
from foamsim.data import reference_curve

# ---------------------------------------------------------------- constituents
matrix = MATERIALS["epoxy"]
particle = hollow_particle("K46")   # eta inferred from 0.46 g/cm3 true density
shell = particle.shell

print("Constituents")
print(f"  matrix   : {matrix.name}  E={matrix.E:.0f} MPa  nu={matrix.nu}  rho={matrix.rho} g/cm3")
print(f"  shell    : {shell.name}  E={shell.E/1000:.0f} GPa  nu={shell.nu}  rho={shell.rho} g/cm3")
print(f"  particle : K46  eta={particle.eta:.4f}  true density={particle.true_density:.3f} g/cm3"
      f"  (datasheet 0.46)")

# ------------------------------------------------------------------ self-checks
e0 = hollow_particle_mori_tanaka(matrix, particle, vf=0.0)
assert abs(e0.E - matrix.E) < 1e-6 * matrix.E, "vf=0 must return the matrix modulus"
assert abs(e0.rho - matrix.rho) < 1e-9, "vf=0 must return the matrix density"
print(f"self-check vf=0 -> E={e0.E:.1f} MPa, rho={e0.rho:.3f} g/cm3 (matrix)  OK")
assert abs(particle.true_density - 0.46) < 1e-9, "K46 true density"

# ------------------------------------------------------------------- vf sweep
rows = []
for vf in np.linspace(0.0, 0.6, 13):
    mt = hollow_particle_mori_tanaka(matrix, particle, vf=vf)
    ds = hollow_particle_differential(matrix, particle, vf=vf)
    hs = hashin_shtrikman_bounds(matrix, particle, vf=vf)
    rows.append({
        "vf": round(float(vf), 4),
        "density_g_cc": density(matrix, particle, vf=vf),
        "E_mt_mpa": mt.E,
        "E_ds_mpa": ds.E,
        "nu_mt": mt.nu,
        "E_hs_lo_mpa": hs["E_lo"],
        "E_hs_hi_mpa": hs["E_hi"],
    })
df = pd.DataFrame(rows)

# every estimate must lie inside the HS band
inside_mt = ((df.E_mt_mpa >= df.E_hs_lo_mpa - 1e-6) & (df.E_mt_mpa <= df.E_hs_hi_mpa + 1e-6)).all()
inside_ds = ((df.E_ds_mpa >= df.E_hs_lo_mpa - 1e-6) & (df.E_ds_mpa <= df.E_hs_hi_mpa + 1e-6)).all()
assert inside_mt and inside_ds, "an estimate fell outside the Hashin-Shtrikman bounds"
print("self-check: HP-MT and HP-DS both inside HS bounds at every vf  OK")

df.to_csv("modulus_vs_vf.csv", index=False)
print("\nmodulus_vs_vf.csv")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

# ------------------------------------------------- experimental comparison
exp = reference_curve("epoxy", "glass_microballoon")
exp = exp[exp.modulus_mpa.notna() & (exp.particle_volume_fraction <= 0.6)].copy()
exp["E_mt_mpa"] = [hollow_particle_mori_tanaka(matrix, particle, vf=v).E
                   for v in exp.particle_volume_fraction]
exp["E_ds_mpa"] = [hollow_particle_differential(matrix, particle, vf=v).E
                   for v in exp.particle_volume_fraction]
hs_exp = [hashin_shtrikman_bounds(matrix, particle, vf=v) for v in exp.particle_volume_fraction]
exp["E_hs_lo_mpa"] = [h["E_lo"] for h in hs_exp]
exp["E_hs_hi_mpa"] = [h["E_hi"] for h in hs_exp]
exp["rel_err_mt_pct"] = 100 * (exp.E_mt_mpa - exp.modulus_mpa) / exp.modulus_mpa
exp["inside_hs_band"] = ((exp.modulus_mpa >= exp.E_hs_lo_mpa) & (exp.modulus_mpa <= exp.E_hs_hi_mpa))

# Grade-matched prediction: where the record reports the particle's own true density, rebuild the
# hollow particle from it instead of assuming K46. (Nothing is tuned; this only uses reported inputs.)
def _matched(row):
    td = row.particle_true_density_g_cc
    if pd.isna(td):
        return np.nan
    pm = HollowParticle.from_true_density(shell, float(td))
    return hollow_particle_mori_tanaka(matrix, pm, vf=float(row.particle_volume_fraction)).E

exp["E_mt_grade_matched_mpa"] = exp.apply(_matched, axis=1)
exp["rel_err_grade_matched_pct"] = (
    100 * (exp.E_mt_grade_matched_mpa - exp.modulus_mpa) / exp.modulus_mpa)
exp.to_csv("experimental_comparison.csv", index=False)

mape_mt = exp.rel_err_mt_pct.abs().mean()
mape_ds = (100 * (exp.E_ds_mpa - exp.modulus_mpa) / exp.modulus_mpa).abs().mean()
print(f"\nExperimental comparison: {len(exp)} FoamGPT epoxy/glass-microballoon quasi-static "
      f"compression rows carrying a modulus")
print(exp[["record_id", "sample_label", "particle_true_density_g_cc", "particle_volume_fraction",
           "measured_density_g_cc", "modulus_mpa", "E_mt_mpa", "rel_err_mt_pct", "inside_hs_band"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print(f"\nMAPE  HP-MT vs experiment: {mape_mt:.0f} %   HP-DS vs experiment: {mape_ds:.0f} %")
print(f"experimental points inside the HS band: {int(exp.inside_hs_band.sum())}/{len(exp)}")

matched = exp[exp.E_mt_grade_matched_mpa.notna()]
if len(matched):
    print(f"\nGrade-matched subset ({len(matched)} row(s) that report their own particle true density):")
    print(matched[["record_id", "particle_true_density_g_cc", "particle_volume_fraction",
                   "modulus_mpa", "E_mt_grade_matched_mpa", "rel_err_grade_matched_pct"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4g}"))

# density cross-check at the vf where most experimental points sit
rho_model_06 = density(matrix, particle, vf=0.6)
rho_meas_06 = exp.loc[exp.particle_volume_fraction.round(2) == 0.60, "measured_density_g_cc"].mean()
print(f"\nDensity cross-check at vf=0.60: model (epoxy + K46) {rho_model_06:.3f} g/cm3 vs measured "
      f"mean {rho_meas_06:.3f} g/cm3 -> the experimental foams are lighter than an epoxy/K46 foam.")

print("""
CAVEAT — read the comparison, do not take the MAPE at face value:
 * The bundled FoamGPT snapshot contains NO monolithic epoxy/K46 compression-modulus series. All 12
   vf=0.60 rows come from ONE study of layered functionally graded syntactic foams (FGSFs) built from
   S22/S32/S38/K46 layers, and each sample appears twice with ~2.5x different moduli (two loading
   orientations relative to the layers). A layered, orientation-dependent FGSF is not the isotropic
   random-dispersion microstructure the Mori-Tanaka / HS theory describes.
 * Those foams use much lighter grades than K46 (S22 = 0.22 g/cm3 vs K46 = 0.46), i.e. thinner walls
   and a more compliant particle; the density cross-check above confirms it.
 * The single grade-matched row (S60HS, true density 0.60, vf=0.30) is the only near-like-for-like
   point and still sits ~2x below the model, consistent with the 20-40 %+ shortfall expected from
   matrix porosity and particle breakage, amplified here by an unreported test standard.
 * Conclusion: the analytical curve is the deliverable; the experimental rows bracket it only in
   order of magnitude and cannot validate it. No constant was tuned and no value was read off a plot.
""")

# --------------------------------------------------------------------- plot
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
ax.fill_between(df.vf, df.E_hs_lo_mpa, df.E_hs_hi_mpa, color="0.85",
                label="Hashin-Shtrikman bounds")
ax.plot(df.vf, df.E_mt_mpa, "-o", ms=4, color="C0", label="hollow-particle Mori-Tanaka")
ax.plot(df.vf, df.E_ds_mpa, "--s", ms=4, color="C1", label="differential scheme")
if len(exp):
    ax.plot(exp.particle_volume_fraction, exp.modulus_mpa, "k^", ms=6, alpha=0.7,
            label=f"experiment, other grades/FGSF (n={len(exp)})")
ax.set_xlabel("particle volume fraction, $v_f$")
ax.set_ylabel("compressive modulus $E$ (MPa)")
ax.set_title("Epoxy / 3M K46 glass microballoons")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax2.plot(df.vf, df.density_g_cc, "-o", ms=4, color="C2", label="model density")
if len(exp) and exp.measured_density_g_cc.notna().any():
    d = exp[exp.measured_density_g_cc.notna()]
    ax2.plot(d.particle_volume_fraction, d.measured_density_g_cc, "k^", ms=6,
             label="measured density")
ax2.set_xlabel("particle volume fraction, $v_f$")
ax2.set_ylabel(r"density (g/cm$^3$)")
ax2.set_title("Density vs volume fraction")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("modulus_vs_vf.png", dpi=160)
print("\nwrote modulus_vs_vf.csv, experimental_comparison.csv, modulus_vs_vf.png")
