---
name: foamsim-fe-rve
description: Numerical (finite-element) homogenization of a random hollow-sphere RVE with foamsim (scikit-fem, voxel hexahedra, KUBC). Use when a numerical check of the analytical estimates or an RVE study is requested.
---

# foamsim FE / RVE — API patterns

```python
from foamsim import MATERIALS, hollow_particle
from foamsim.rve import random_packing
from foamsim.fem import homogenize, homogenize_homogeneous, ResolutionError

p = hollow_particle("K46"); m = MATERIALS["epoxy"]
rve = random_packing(vf=0.3, n_spheres=16, eta=p.eta, seed=0)   # periodic RSA packing; rve.vf is the realised fraction
eff = homogenize(rve, m, p, n=24, mode="equivalent")             # Effective; model="FE-KUBC-equivalent-n24"
```

- `mode="equivalent"` (default): each hollow sphere is a homogeneous equivalent particle; fast (n=24 -> ~10-30 s CPU).
- `mode="shell"`: explicit glass shell + void core. Requires >= 2 voxels across the wall or raises `ResolutionError`;
  for eta ~ 0.94 and radius ~0.1 box this needs n >= ~64 (minutes). Do not silently drop to "equivalent" — report it.
- KUBC (displacement BCs) overestimates stiffness for small RVEs; average >= 2 seeds and report the spread.
- `random_packing` supports vf < ~0.55 with 12-30 spheres; higher vf raises. Use the realised `rve.vf`, not the target.
- Sanity check: `homogenize_homogeneous(m, n=4)` must return the matrix moduli exactly.
- Compare FE with `hollow_particle_mori_tanaka(m, p, rve.vf)` and the HS bounds; FE outside the bounds means a bug.
