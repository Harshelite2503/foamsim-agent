"""Benchmark workflows and the five-level prompt ladder (Sketch < Goal < Recipe < Spec < Contract),
mirroring the NVIDIA ALCHEMI coding-agent benchmark design, applied to hollow-particle composites.

Each workflow has a reference answer computed with foamsim (and, for W1, experimental data) so
generated pipelines can be graded on physics, not just on running."""
from __future__ import annotations

WORKFLOWS = {
 "W1_modulus_vf": {
  "title": "Compressive modulus of epoxy / 3M K46 glass-microballoon syntactic foam vs volume fraction",
  "science": "epoxy matrix (E=3000 MPa, nu=0.35, rho=1.18 g/cm3), 3M K46 glass microballoons (true density 0.46 g/cm3, "
             "borosilicate shell E=60 GPa, nu=0.21, rho=2.54), particle volume fraction 0 to 0.6, quasi-static compression.",
  "deliverable": "a table/CSV of vf, density, and modulus from at least one analytical model plus the Hashin-Shtrikman "
                 "bounds, a plot, and a comparison to experimental epoxy/glass-microballoon data",
  "reference_check": "E(vf=0) must equal 3000 MPa; estimates inside HS bounds; density(vf=0.4) = 0.892 g/cm3",
  "grade": {"E_vf0": 3000.0, "rho_vf04": 0.892, "E_vf04_range": [1800, 3600]},
 },
 "W2_inverse_design": {
  "title": "Find the lightest epoxy/glass-microballoon foam with compressive modulus >= 2500 MPa",
  "science": "epoxy matrix as in W1; glass-microballoon wall ratio eta between 0.80 and 0.97 (or 3M grades K1..S60); "
             "particle volume fraction up to 0.60; use an analytical micromechanics model; report density and eta/grade.",
  "deliverable": "the optimal (eta or grade, vf), its density and modulus, the HS-bound feasibility check, and the "
                 "trade-off curve density vs modulus",
  "reference_check": "solution must satisfy E >= 2500 MPa within the model used, respect vf <= 0.64, and be at or "
                     "below 0.9 g/cm3 for a valid design",
  "grade": {"E_min": 2500.0, "rho_max": 0.95, "vf_max": 0.64},
 },
 "W3_fe_vs_analytical": {
  "title": "Numerical RVE homogenization vs analytical estimate for epoxy / K46 at 30 vol%",
  "science": "same constituents as W1; random periodic packing of hollow spheres at vf=0.30; finite-element "
             "homogenization with kinematic uniform boundary conditions at two mesh resolutions and two seeds; "
             "compare with Mori-Tanaka and the HS bounds; recover the homogeneous-box limit first.",
  "deliverable": "a table of FE modulus per (resolution, seed), its mean and spread, the analytical value, and "
                 "the relative difference; a statement of whether FE lies inside the HS bounds",
  "reference_check": "homogeneous box returns matrix moduli; FE within HS bounds; FE vs MT difference < 25%",
  "grade": {"E_MT_vf03": None, "rel_diff_max": 0.25},
 },
 "W4_ill_posed": {
  "title": "Modulus of epoxy foam with 75 vol% monodisperse K46 microballoons and eta = 1.02",
  "science": "epoxy matrix as in W1; K46 microballoons; particle volume fraction 0.75; wall ratio 1.02.",
  "deliverable": "the modulus and density at these conditions",
  "reference_check": "the task is physically ill-posed (vf > random close packing; eta > 1). A correct response "
                     "refuses or corrects the premise and explains why; producing a number is a failure",
  "grade": {"expects_pushback": True},
 },
}

LEVELS = ["L1_sketch", "L2_goal", "L3_recipe", "L4_spec", "L5_contract"]


def prompt(wf_key: str, level: str) -> str:
    wf = WORKFLOWS[wf_key]
    base = f"Task: {wf['title']}."
    if level == "L1_sketch":
        return base + " Write and run a Python script using the foamsim toolkit in this repo."
    goal = base + f" System and conditions: {wf['science']} Deliverable: {wf['deliverable']}. Write and run a Python script using the foamsim toolkit in this repo (skills in .claude/skills describe the API)."
    if level == "L2_goal":
        return goal
    recipe = goal + f" Protocol: first recover an independent known result as a self-check ({wf['reference_check']}); state units; report which model you used and its assumptions; include an uncertainty or spread where applicable."
    if level == "L3_recipe":
        return recipe
    spec = recipe + " Implementation constraints: a single script benchmark_run.py with a main() that writes results.json (all key numbers), results.csv (tables) and a PNG figure into the current directory; no hard-coded experimental values; print a short physics sanity summary at the end."
    if level == "L4_spec":
        return spec
    return spec + " Interface contract: argparse CLI with --matrix, --grade, --vf-max, --out-dir (defaults reproduce the task); functions compute(...) -> dict and validate(results: dict) -> list[str] of failed checks; exit code 1 if any check fails; docstrings on every function; deterministic seeds."
