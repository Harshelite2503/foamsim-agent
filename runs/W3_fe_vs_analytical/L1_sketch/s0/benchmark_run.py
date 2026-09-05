"""Numerical RVE (FE) homogenization vs analytical estimates: epoxy / K46 at 30 vol%.

Steps
  1. Sanity check: FE on a homogeneous box must return the matrix moduli.
  2. Analytical references: hollow-particle Mori-Tanaka, differential scheme, Hashin-Shtrikman bounds.
  3. FE voxel homogenization (scikit-fem, KUBC, equivalent-particle mode) on random hollow-sphere
     packings, 2 seeds, n = 24; report mean and spread.
  4. Compare: FE vs MT relative difference, and whether FE lies inside the HS band.
"""

import json

import numpy as np

from foamsim import MATERIALS, hollow_particle
from foamsim.fem import homogenize, homogenize_homogeneous
from foamsim.micromechanics import (
    density,
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
)
from foamsim.rve import random_packing

VF = 0.30
N = 24
N_SPHERES = 16
SEEDS = [0, 1]

matrix = MATERIALS["epoxy"]
particle = hollow_particle("K46")

print("=== Setup ===")
print(f"matrix   : epoxy   E = {matrix.E:.1f} MPa, nu = {matrix.nu:.3f}, rho = {matrix.rho:.3f} g/cc")
print(f"particle : K46     eta = {particle.eta:.4f}, true density = {particle.true_density:.3f} g/cc, "
      f"d = {particle.diameter_um} um")
print(f"target particle volume fraction (incl. hollow cores) = {VF:.2f}")

# --- 1. homogeneous-box limit -------------------------------------------------
print("\n=== 1. FE sanity check: homogeneous box ===")
hom = homogenize_homogeneous(matrix, n=4)
err_E = abs(hom.E - matrix.E) / matrix.E
err_nu = abs(hom.nu - matrix.nu)
print(f"FE homogeneous box : E = {hom.E:.2f} MPa (matrix {matrix.E:.2f}), rel. err = {err_E:.2e}")
print(f"                     nu = {hom.nu:.5f} (matrix {matrix.nu:.5f}), abs. err = {err_nu:.2e}")
assert err_E < 1e-6 and err_nu < 1e-6, "FE does not recover the homogeneous limit"
print("PASS")

# --- 2. analytical references -------------------------------------------------
print("\n=== 2. Analytical estimates at vf = 0.30 ===")
mt = hollow_particle_mori_tanaka(matrix, particle, VF)
ds = hollow_particle_differential(matrix, particle, VF)
hs = hashin_shtrikman_bounds(matrix, particle, VF)
rho = density(matrix, particle, VF)
print(f"Mori-Tanaka (HP-MT)  : E = {mt.E:9.1f} MPa, K = {mt.K:9.1f}, G = {mt.G:9.1f}, nu = {mt.nu:.4f}")
print(f"Differential (HP-DS) : E = {ds.E:9.1f} MPa, K = {ds.K:9.1f}, G = {ds.G:9.1f}, nu = {ds.nu:.4f}")
print(f"HS bounds            : E in [{hs['E_lo']:.1f}, {hs['E_hi']:.1f}] MPa, "
      f"K in [{hs['K_lo']:.1f}, {hs['K_hi']:.1f}] MPa")
print(f"density (rule of mixtures) = {rho:.4f} g/cc  (matrix {matrix.rho:.3f})")
assert hs["E_lo"] <= mt.E <= hs["E_hi"], "MT outside HS bounds"

# --- 3. FE RVE homogenization -------------------------------------------------
print("\n=== 3. FE RVE homogenization (KUBC, equivalent particle, n = 24) ===")
rows = []
for seed in SEEDS:
    rve = random_packing(vf=VF, n_spheres=N_SPHERES, eta=particle.eta, seed=seed)
    eff = homogenize(rve, matrix, particle, n=N, mode="equivalent")
    vox = rve.voxelize(N)
    vf_vox = float((vox > 0).mean())
    rows.append({"seed": seed, "vf_realised": rve.vf, "vf_voxelised": vf_vox,
                 "E_mpa": eff.E, "K_mpa": eff.K, "G_mpa": eff.G, "nu": eff.nu, "model": eff.model})
    print(f"seed {seed}: vf(analytic) = {rve.vf:.4f}, vf(voxelised at n={N}) = {vf_vox:.4f} -> "
          f"E = {eff.E:.1f} MPa, K = {eff.K:.1f}, G = {eff.G:.1f}, nu = {eff.nu:.4f}")

E_fe = np.array([r["E_mpa"] for r in rows])
E_mean, E_std = float(E_fe.mean()), float(E_fe.std(ddof=1))
print(f"FE mean over {len(SEEDS)} seeds: E = {E_mean:.1f} +/- {E_std:.1f} MPa "
      f"(spread {2 * E_std / E_mean * 100:.1f} % of mean)")

# --- 4. comparison ------------------------------------------------------------
print("\n=== 4. FE vs analytical ===")
d_mt = (E_mean - mt.E) / mt.E * 100
d_ds = (E_mean - ds.E) / ds.E * 100
inside = hs["E_lo"] <= E_mean <= hs["E_hi"]
print(f"E_FE / E_MT  = {E_mean / mt.E:.4f}   ({d_mt:+.1f} %)")
print(f"E_FE / E_DS  = {E_mean / ds.E:.4f}   ({d_ds:+.1f} %)")
print(f"E_FE inside HS bounds [{hs['E_lo']:.1f}, {hs['E_hi']:.1f}] : {inside}")
print(f"agreement with MT within 25 % : {abs(d_mt) < 25.0}")
print("Note: KUBC on a small RVE (16 spheres) is a stiff (upper) estimate, so FE >= MT is expected.")

summary = {
    "matrix": "epoxy", "grade": "K46", "vf_target": VF, "n_voxels": N, "n_spheres": N_SPHERES,
    "seeds": SEEDS, "eta": particle.eta,
    "homogeneous_box_rel_err_E": err_E,
    "E_mt_mpa": mt.E, "E_ds_mpa": ds.E, "hs_E_lo_mpa": hs["E_lo"], "hs_E_hi_mpa": hs["E_hi"],
    "density_g_cc": rho,
    "fe_runs": rows, "E_fe_mean_mpa": E_mean, "E_fe_std_mpa": E_std,
    "fe_vs_mt_pct": d_mt, "fe_vs_ds_pct": d_ds, "fe_inside_hs_bounds": bool(inside),
}
with open("results.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nwrote results.json")
