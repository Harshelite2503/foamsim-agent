---
name: foamsim-micromechanics
description: Analytical effective properties (modulus, density, HS bounds, crush onset) of hollow-particle / syntactic-foam composites with the foamsim toolkit. Use for any modulus-vs-volume-fraction, wall-thickness (eta) or density calculation.
---

# foamsim micromechanics — API patterns

```python
from foamsim import MATERIALS, hollow_particle
from foamsim.materials import Isotropic, HollowParticle
from foamsim.micromechanics import (density, hollow_particle_mori_tanaka, hollow_particle_differential,
                                    hashin_shtrikman_bounds, hollow_sphere_equivalent, gibson_ashby,
                                    particle_crush_onset, RCP)

m = MATERIALS["epoxy"]                  # Isotropic(E=3000 MPa, nu=0.35, rho=1.18)
p = hollow_particle("K46")              # 3M glass bubble, true density 0.46 -> eta inferred (~0.937)
p2 = HollowParticle(MATERIALS["glass"], eta=0.95, diameter_um=40)   # explicit wall ratio
p3 = HollowParticle.from_true_density(MATERIALS["glass"], 0.38)      # from datasheet density

e = hollow_particle_mori_tanaka(m, p, vf=0.4)     # Effective(K, G, rho, vf, model); e.E, e.nu, e.as_dict()
d = hollow_particle_differential(m, p, vf=0.4)    # differential scheme; usually slightly below MT at high vf
b = hashin_shtrikman_bounds(m, p, vf=0.4)         # dict E_lo/E_hi/K_lo/...; every estimate must lie inside
rho = density(m, p, vf=0.4, matrix_porosity=0.02) # g/cm^3
sigma_c = particle_crush_onset(p, m, vf=0.4)      # MPa, order of magnitude only
```

Units: MPa, g/cm^3, micrometres. vf is the volume fraction of particles INCLUDING their hollow cores.
`vf > RCP (0.64)` raises ValueError (not realisable for monodisperse spheres). `eta` must be in [0,1).

Conventions
- Modulus = Young's modulus of the composite; for syntactic foams the compressive modulus is what papers report.
- Stiff particles (K46, S60) raise E above the matrix; light particles (K1, K15) lower it — check the sign.
- Sweeps: loop vf in `np.linspace(0, 0.6, 13)`; collect `e.as_dict()` rows into a DataFrame.
- Always report which model (HP-MT vs HP-DS) and show the HS band; the models are estimates, not truth.

Reference limits (useful self-checks): vf=0 -> matrix; eta=0 -> Mori-Tanaka solid spheres; both estimates inside HS bounds.
