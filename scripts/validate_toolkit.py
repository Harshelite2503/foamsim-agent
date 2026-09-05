"""Toolkit validation: FE (equivalent-particle, KUBC) vs HP-MT / HP-DS / HS bounds, and vs FoamGPT
experimental epoxy/glass-microballoon data. Writes data/validation.csv and data/validation.png."""
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg"); import matplotlib.pyplot as plt
from foamsim import MATERIALS, hollow_particle
from foamsim.data import reference_curve
from foamsim.fem import homogenize
from foamsim.micromechanics import hashin_shtrikman_bounds, hollow_particle_differential, hollow_particle_mori_tanaka
from foamsim.rve import random_packing

EP = MATERIALS["epoxy"]; rows = []
for grade in ("K46", "S38", "S22"):
    p = hollow_particle(grade)
    for vf in (0.1, 0.2, 0.3, 0.4, 0.5):
        mt = hollow_particle_mori_tanaka(EP, p, vf); ds = hollow_particle_differential(EP, p, vf); b = hashin_shtrikman_bounds(EP, p, vf)
        r = {"grade": grade, "eta": p.eta, "vf": vf, "E_MT": mt.E, "E_DS": ds.E, "E_HS_lo": b["E_lo"], "E_HS_hi": b["E_hi"], "rho": mt.rho}
        if grade == "K46":
            t = time.time(); fe = [homogenize(random_packing(vf, n_spheres=16, eta=p.eta, seed=s), EP, p, n=24) for s in (0, 1)]
            r.update({"E_FE_mean": np.mean([f.E for f in fe]), "E_FE_std": np.std([f.E for f in fe]), "vf_FE": fe[0].vf, "fe_seconds": (time.time() - t) / 2})
        rows.append(r); print(r)
df = pd.DataFrame(rows); df.to_csv("data/validation.csv", index=False)
ref = reference_curve("epoxy", "glass_microballoon"); ref = ref[ref.modulus_mpa.notna()]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for grade, g in df.groupby("grade"):
    ax.plot(g.vf, g.E_MT, "-", label=f"HP-MT {grade} (η={g.eta.iloc[0]:.2f})"); ax.plot(g.vf, g.E_DS, "--", color=ax.lines[-1].get_color(), alpha=.7)
    ax.fill_between(g.vf, g.E_HS_lo, g.E_HS_hi, color=ax.lines[-1].get_color(), alpha=.08)
k = df[df.grade == "K46"]; ax.errorbar(k.vf, k.E_FE_mean, yerr=k.E_FE_std, fmt="ks", ms=5, label="FE KUBC n=24 (K46)")
ax.scatter(ref.particle_volume_fraction, ref.modulus_mpa, marker="x", color="tab:red", s=28, label=f"FoamGPT epoxy/GMB data (n={len(ref)})")
ax.set_xlabel("particle volume fraction"); ax.set_ylabel("compressive modulus (MPa)"); ax.set_title("Toolkit validation: analytical vs FE vs experiment"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("data/validation.png", dpi=200); print("saved; FoamGPT ref rows:", len(ref))
