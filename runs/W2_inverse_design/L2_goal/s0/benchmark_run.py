"""W2 inverse design: lightest epoxy / glass-microballoon syntactic foam with E_c >= 3500 MPa.

Design space
    eta  (wall ratio r_in/r_out) in [0.80, 0.97]      -- continuous sweep + the 3M grades K1..S60
    vf   (particle volume fraction, cores included) in [0, 0.60]

Objective   minimise composite density (g/cm^3)
Constraint  compressive Young's modulus >= 3500 MPa

Model: analytical micromechanics from the foamsim toolkit --
    HP-MT  hollow_particle_mori_tanaka  (equivalent hollow-sphere particle -> Mori-Tanaka)
    HP-DS  hollow_particle_differential (cross-check)
    Hashin-Shtrikman bounds are computed for every candidate; the estimate must lie inside,
    and the HS upper bound tells us whether the target is attainable at all.

Outputs: printed report + results.csv (full grid) + tradeoff.csv (Pareto front) + tradeoff.png
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
    RCP,
    density,
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
    hollow_sphere_equivalent,
)

TARGET_E = 3500.0          # MPa, compressive modulus requirement
ETA_MIN, ETA_MAX = 0.80, 0.97
VF_MAX = 0.60
GRADES = ["K1", "K15", "K20", "S22", "K25", "S32", "S38", "K46", "S60"]

matrix = MATERIALS["epoxy"]
glass = MATERIALS["glass"]


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# ---------------------------------------------------------------- 0. sanity checks
banner("0. Reference limits (self-checks)")
e0 = hollow_particle_mori_tanaka(matrix, HollowParticle(glass, eta=0.90), vf=0.0)
print(f"vf=0 -> E = {e0.E:9.2f} MPa (matrix E = {matrix.E:.2f})   rho = {e0.rho:.4f} "
      f"(matrix rho = {matrix.rho:.4f})")
assert abs(e0.E - matrix.E) < 1e-6 * matrix.E and abs(e0.rho - matrix.rho) < 1e-9
print(f"vf cap: task asks vf <= {VF_MAX}; toolkit packing limit RCP = {RCP} -> "
      f"{'OK, inside packing limit' if VF_MAX <= RCP else 'EXCEEDS packing limit'}")
print("note: vf > ~0.55 in practice requires polydisperse packing.")
print(f"eta window [{ETA_MIN}, {ETA_MAX}] is inside the physical range [0, 1).")


# ---------------------------------------------------------------- 1. feasibility vs HS upper bound
banner("1. Feasibility: is E >= 3500 MPa reachable inside the design space?")
# The stiffest admissible microstructure is the thickest-wall particle (eta = ETA_MIN)
# at the largest volume fraction (vf = VF_MAX). Its HS upper bound is the ceiling for
# *any* microstructure with these constituents and that composition.
p_stiff = HollowParticle(glass, eta=ETA_MIN)
b_stiff = hashin_shtrikman_bounds(matrix, p_stiff, VF_MAX)
e_stiff = hollow_particle_mori_tanaka(matrix, p_stiff, VF_MAX)
print(f"stiffest corner: eta = {ETA_MIN}, vf = {VF_MAX}")
print(f"  equivalent particle: {hollow_sphere_equivalent(p_stiff).name}, "
      f"E_p = {hollow_sphere_equivalent(p_stiff).E:.1f} MPa, "
      f"rho_p = {p_stiff.true_density:.4f} g/cm^3")
print(f"  HS bounds  E_lo = {b_stiff['E_lo']:8.1f} MPa   E_hi = {b_stiff['E_hi']:8.1f} MPa")
print(f"  HP-MT      E    = {e_stiff.E:8.1f} MPa   rho = {e_stiff.rho:.4f} g/cm^3")
feasible_bound = b_stiff["E_hi"] >= TARGET_E
print(f"  target {TARGET_E:.0f} MPa vs HS upper bound {b_stiff['E_hi']:.1f} MPa -> "
      f"{'ATTAINABLE in principle' if feasible_bound else 'IMPOSSIBLE for any microstructure'}")


# ---------------------------------------------------------------- 2. grid sweep
banner("2. Design-space sweep (HP-MT, with HP-DS and HS bounds)")
etas = np.round(np.arange(ETA_MIN, ETA_MAX + 1e-9, 0.001), 4)
vfs = np.round(np.arange(0.0, VF_MAX + 1e-9, 0.01), 4)

rows = []
for eta in etas:
    p = HollowParticle(glass, eta=float(eta))
    for vf in vfs:
        mt = hollow_particle_mori_tanaka(matrix, p, float(vf))
        ds = hollow_particle_differential(matrix, p, float(vf))
        b = hashin_shtrikman_bounds(matrix, p, float(vf))
        rows.append({
            "eta": float(eta), "vf": float(vf),
            "rho_g_cc": mt.rho,
            "E_mt_mpa": mt.E, "E_ds_mpa": ds.E,
            "E_hs_lo": b["E_lo"], "E_hs_hi": b["E_hi"],
            "nu_mt": mt.nu,
            "particle_true_density": p.true_density,
            "inside_hs_mt": b["E_lo"] - 1e-6 <= mt.E <= b["E_hi"] + 1e-6,
            "inside_hs_ds": b["E_lo"] - 1e-6 <= ds.E <= b["E_hi"] + 1e-6,
        })
df = pd.DataFrame(rows)
df["meets_target_mt"] = df["E_mt_mpa"] >= TARGET_E
df["meets_target_ds"] = df["E_ds_mpa"] >= TARGET_E
print(f"grid: {len(etas)} eta values x {len(vfs)} vf values = {len(df)} points")
print(f"all HP-MT estimates inside HS bounds: {bool(df['inside_hs_mt'].all())}")
print(f"all HP-DS estimates inside HS bounds: {bool(df['inside_hs_ds'].all())}")
print(f"points meeting E >= {TARGET_E:.0f} MPa (HP-MT): {int(df['meets_target_mt'].sum())} "
      f"/ {len(df)}")
df.to_csv("results.csv", index=False)


# ---------------------------------------------------------------- 3. optimum over continuous eta
banner("3. Optimum: lightest feasible design (continuous eta)")
feas = df[df["meets_target_mt"]].copy()
if feas.empty:
    print("NO feasible design in the stated box -- target cannot be met.")
    best = None
else:
    best = feas.sort_values(["rho_g_cc", "E_mt_mpa"], ascending=[True, False]).iloc[0]
    p_best = HollowParticle(glass, eta=float(best["eta"]))
    b_best = hashin_shtrikman_bounds(matrix, p_best, float(best["vf"]))
    print(f"  eta*                 = {best['eta']:.3f}  "
          f"(wall t/R = {1 - best['eta']:.3f}, particle true density "
          f"{p_best.true_density:.4f} g/cm^3)")
    print(f"  vf*                  = {best['vf']:.3f}")
    print(f"  density              = {best['rho_g_cc']:.4f} g/cm^3   "
          f"({100 * (1 - best['rho_g_cc'] / matrix.rho):.1f} % lighter than neat epoxy)")
    print(f"  E (HP-MT)            = {best['E_mt_mpa']:.1f} MPa   (target {TARGET_E:.0f})")
    print(f"  E (HP-DS cross-check)= {best['E_ds_mpa']:.1f} MPa "
          f"({'meets' if best['E_ds_mpa'] >= TARGET_E else 'MISSES'} target)")
    print(f"  HS band at this point: [{b_best['E_lo']:.1f}, {b_best['E_hi']:.1f}] MPa -> "
          f"estimate inside bounds: {b_best['E_lo'] <= best['E_mt_mpa'] <= b_best['E_hi']}")
    print(f"  nu                   = {best['nu_mt']:.4f}")
    print(f"  specific modulus     = {best['E_mt_mpa'] / best['rho_g_cc']:.1f} MPa/(g/cm^3)")

# why the optimum sits where it does
if best is not None:
    at_vfmax = df[np.isclose(df["vf"], VF_MAX)]
    print(f"\n  (the optimum sits at vf = {best['vf']:.2f}; the vf ceiling in the task is "
          f"{VF_MAX} and lighter, thinner-walled particles trade modulus for density)")
    print(f"  lightest design at vf = {VF_MAX} that still meets target: "
          f"{at_vfmax[at_vfmax['meets_target_mt']]['rho_g_cc'].min() if at_vfmax['meets_target_mt'].any() else float('nan'):.4f} g/cm^3")


# ---------------------------------------------------------------- 4. 3M grades
banner("4. Same search restricted to 3M glass-bubble grades")
grade_rows = []
for g in GRADES:
    p = hollow_particle(g)
    in_window = ETA_MIN <= p.eta <= ETA_MAX
    for vf in vfs:
        mt = hollow_particle_mori_tanaka(matrix, p, float(vf))
        b = hashin_shtrikman_bounds(matrix, p, float(vf))
        grade_rows.append({"grade": g, "eta": p.eta, "eta_in_window": in_window,
                           "vf": float(vf), "rho_g_cc": mt.rho, "E_mt_mpa": mt.E,
                           "E_hs_lo": b["E_lo"], "E_hs_hi": b["E_hi"]})
gdf = pd.DataFrame(grade_rows)
gdf["meets_target"] = gdf["E_mt_mpa"] >= TARGET_E

print(f"{'grade':>6} {'eta':>7} {'in [0.80,0.97]':>15} {'rho_p':>7} "
      f"{'min vf meeting target':>22} {'rho at that vf':>15}")
for g in GRADES:
    sub = gdf[gdf["grade"] == g]
    p = hollow_particle(g)
    ok = sub[sub["meets_target"]]
    if ok.empty:
        vf_s, rho_s = "none <= 0.60", "-"
    else:
        r = ok.sort_values("vf").iloc[0]
        vf_s, rho_s = f"{r['vf']:.2f}", f"{r['rho_g_cc']:.4f}"
    print(f"{g:>6} {p.eta:7.4f} {str(bool(sub['eta_in_window'].iloc[0])):>15} "
          f"{p.true_density:7.3f} {vf_s:>22} {rho_s:>15}")

gfeas = gdf[gdf["meets_target"] & gdf["eta_in_window"]]
if gfeas.empty:
    print("\nno in-window grade meets the target at vf <= 0.60")
    gbest = None
else:
    gbest = gfeas.sort_values(["rho_g_cc", "E_mt_mpa"], ascending=[True, False]).iloc[0]
    print(f"\nbest grade design: {gbest['grade']} (eta = {gbest['eta']:.4f}) at vf = {gbest['vf']:.2f}")
    print(f"  density = {gbest['rho_g_cc']:.4f} g/cm^3, E = {gbest['E_mt_mpa']:.1f} MPa, "
          f"HS band [{gbest['E_hs_lo']:.1f}, {gbest['E_hs_hi']:.1f}] MPa")
gdf.to_csv("grades.csv", index=False)


# ---------------------------------------------------------------- 5. trade-off curve
banner("5. Trade-off curve: minimum achievable density vs required modulus")
targets = np.arange(3000.0, 4201.0, 25.0)
tr = []
for t in targets:
    sub = df[df["E_mt_mpa"] >= t]
    if sub.empty:
        tr.append({"E_required_mpa": t, "min_density_g_cc": np.nan, "eta": np.nan, "vf": np.nan})
        continue
    r = sub.sort_values("rho_g_cc").iloc[0]
    tr.append({"E_required_mpa": t, "min_density_g_cc": r["rho_g_cc"], "eta": r["eta"],
               "vf": r["vf"], "E_achieved_mpa": r["E_mt_mpa"]})
tdf = pd.DataFrame(tr)
tdf.to_csv("tradeoff.csv", index=False)
show = tdf.dropna(subset=["min_density_g_cc"])
print(f"{'E_req (MPa)':>12} {'min rho (g/cm^3)':>18} {'eta':>7} {'vf':>6}")
for _, r in show.iloc[::4].iterrows():
    print(f"{r['E_required_mpa']:12.0f} {r['min_density_g_cc']:18.4f} {r['eta']:7.3f} {r['vf']:6.2f}")
if not show.empty:
    print(f"highest modulus reachable in the box (HP-MT): {df['E_mt_mpa'].max():.1f} MPa "
          f"at eta = {df.loc[df['E_mt_mpa'].idxmax(), 'eta']:.3f}, "
          f"vf = {df.loc[df['E_mt_mpa'].idxmax(), 'vf']:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for eta in [0.80, 0.85, 0.90, 0.935, 0.97]:
    sub = df[np.isclose(df["eta"], eta)]
    if not sub.empty:
        ax[0].plot(sub["rho_g_cc"], sub["E_mt_mpa"], lw=1.2, label=f"eta = {eta:.3f}")
ax[0].axhline(TARGET_E, color="k", ls="--", lw=1, label=f"target {TARGET_E:.0f} MPa")
if best is not None:
    ax[0].plot(best["rho_g_cc"], best["E_mt_mpa"], "r*", ms=14, label="optimum")
ax[0].set_xlabel("density (g/cm$^3$)"); ax[0].set_ylabel("E, HP-MT (MPa)")
ax[0].set_title("modulus vs density, epoxy / glass microballoons")
ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

ax[1].plot(show["E_required_mpa"], show["min_density_g_cc"], "b-", lw=1.6)
ax[1].axvline(TARGET_E, color="k", ls="--", lw=1)
if best is not None:
    ax[1].plot(TARGET_E, best["rho_g_cc"], "r*", ms=14)
ax[1].set_xlabel("required modulus (MPa)"); ax[1].set_ylabel("minimum achievable density (g/cm$^3$)")
ax[1].set_title("Pareto front: lightest foam meeting a modulus target")
ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig("tradeoff.png", dpi=150)
print("wrote results.csv, grades.csv, tradeoff.csv, tradeoff.png")


# ---------------------------------------------------------------- 6. summary
banner("6. Answer")
if best is None:
    print(f"E >= {TARGET_E:.0f} MPa is not achievable for eta in [{ETA_MIN},{ETA_MAX}], vf <= {VF_MAX}.")
else:
    print(f"optimum (continuous eta): eta = {best['eta']:.3f}, vf = {best['vf']:.2f}")
    print(f"  density {best['rho_g_cc']:.4f} g/cm^3, E(HP-MT) {best['E_mt_mpa']:.1f} MPa, "
          f"E(HP-DS) {best['E_ds_mpa']:.1f} MPa")
    print(f"  HS feasibility: target is below the HS upper bound "
          f"({b_stiff['E_hi']:.1f} MPa at the stiffest corner) -> not bound-violating; "
          f"the HP-MT estimate lies inside its own HS band.")
if gbest is not None:
    print(f"nearest catalogue grade: {gbest['grade']} at vf = {gbest['vf']:.2f} -> "
          f"rho {gbest['rho_g_cc']:.4f} g/cm^3, E {gbest['E_mt_mpa']:.1f} MPa")
print("\nCaveat: HP-MT/HP-DS are idealised estimates (perfect bonding, no matrix porosity, "
      "no particle breakage); experimental syntactic-foam moduli typically fall 20-40 % below "
      "HP-MT, so the design should carry margin above the 3500 MPa target.")
