---
name: foamsim-validate
description: Compare foamsim predictions with experimental syntactic-foam data (FoamGPT dataset bundled) and run premise checks. Use whenever a result should be validated against experiment or a task may be physically ill-posed.
---

# foamsim validation — API patterns

```python
from foamsim.data import load_foamgpt, reference_curve
ref = reference_curve("epoxy", "glass_microballoon")   # primary, unflagged, quasi-static compression rows with vf
ref[["particle_grade", "particle_volume_fraction", "measured_density_g_cc", "modulus_mpa", "strength_mpa"]]
```
matrix_class values: epoxy, vinyl_ester, polyurethane, hdpe, aluminum, ... ; particle_type: glass_microballoon,
fly_ash_cenosphere, ceramic_hollow, polymer_hollow.

Validation protocol
1. Recover an independent known result first (vf=0 -> matrix; HS bounds contain the estimate; homogeneous FE box).
2. Compare predictions with `reference_curve` rows of the same matrix class; report MAPE and whether experimental
   points fall inside the HS band. Experimental moduli are often 20-40% below HP-MT because of matrix porosity and
   particle breakage — say so rather than tuning constants silently.
3. Premise checks the agent must perform and state explicitly:
   - vf > 0.64 is not realisable for monodisperse spheres; vf > ~0.55 needs polydisperse packing.
   - eta >= 1 or true density > shell density is impossible.
   - a target modulus above the HS upper bound for the given constituents cannot be met by any microstructure.
   - metal-matrix foams reported with "modulus" of tens of MPa are ISO 13314 structural stiffness in GPa, not E.
4. Never read values off plots; never hard-code an experimental number to make a check pass.
