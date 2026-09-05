# FoamSim-Agent

**Can AI coding agents run trustworthy micromechanics simulations of hollow-particle composites?**

Inspired by the NVIDIA ALCHEMI coding-agent benchmark for atomistic simulation, applied to a field where the
ground truth is *experimental*: syntactic foams (hollow microspheres in a polymer or metal matrix). Three parts:

1. **`foamsim`** — a small, validated, composable simulation toolkit  
   analytical homogenization (Hashin–Shtrikman bounds, hollow-particle Mori–Tanaka and differential scheme built on
   Hashin's exact hollow-sphere bulk modulus, Gibson–Ashby), density, microballoon crush onset, random hollow-sphere
   RVE packing, voxel finite-element homogenization (scikit-fem, KUBC), and a bridge to the FoamGPT experimental dataset.
2. **Agent skills** (`.claude/skills/`) — API patterns and validation protocol the agent loads on demand.
3. **Benchmark** — 4 workflows × 5 prompt-specificity levels (Sketch → Contract), agent-generated pipelines graded on
   code features, execution, and physics agreement with analytical references *and* experimental data, including an
   ill-posed task that tests whether the agent pushes back.

```
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
pytest -q                                  # 13 tests: limits, HS bounds, FE homogeneous box, packing
foamsim estimate --matrix epoxy --grade K46 --vf 0.4
foamsim fe --matrix epoxy --grade K46 --vf 0.3 --n 24
python scripts/validate_toolkit.py         # FE vs analytical vs FoamGPT data -> data/validation.png
python -m benchmark.prompts                # the 20 benchmark prompts
python -m benchmark.grade                  # grade runs/<wf>/<level>/<sample>/
```

## Toolkit API (30 seconds)
```python
from foamsim import MATERIALS, hollow_particle
from foamsim.micromechanics import hollow_particle_mori_tanaka, hashin_shtrikman_bounds, density
m, p = MATERIALS["epoxy"], hollow_particle("K46")          # eta inferred from 0.46 g/cm3 true density
e = hollow_particle_mori_tanaka(m, p, vf=0.4)               # e.E (MPa), e.nu, e.rho
b = hashin_shtrikman_bounds(m, p, 0.4)                      # every estimate must lie in [E_lo, E_hi]
```

## Workflows
| id | task | reference |
|---|---|---|
| W1 | modulus vs volume fraction, epoxy/K46 | E(0)=matrix, HS bounds, density rule of mixtures, FoamGPT data |
| W2 | lightest foam with E ≥ 2500 MPa (inverse design over η, vf) | feasibility vs HS bound, packing limit |
| W3 | FE RVE homogenization vs Mori–Tanaka | homogeneous-box limit, FE inside HS bounds, < 25 % from MT |
| W4 | ill-posed: 75 vol % monodisperse spheres, η = 1.02 | correct answer is to push back |

Prompt levels add, cumulatively: L1 sketch · L2 system + deliverable · L3 protocol/self-check · L4 file contract ·
L5 CLI/function interface contract.

## Status
- [x] toolkit + tests + skills + validation script
- [x] 20-cell benchmark run: 20/20 executed, 20/20 physics-pass, 5/5 ill-posed runs pushed back (toolkit guards)
- [x] draft paper: `FoamSim_Agent_Draft_Paper.docx`

MIT. Harsh Vardhan Gupta; collaboration with Prof. Nikhil Gupta (NYU Tandon).
