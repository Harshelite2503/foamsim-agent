"""foamsim CLI: quick estimates and sweeps.
    foamsim estimate --matrix epoxy --grade K46 --vf 0.4
    foamsim sweep --matrix epoxy --grade K46 --out sweep.csv
    foamsim fe --matrix epoxy --grade K46 --vf 0.3 --n 24
"""
from __future__ import annotations

import json

import typer
from rich.console import Console

from foamsim import MATERIALS, hollow_particle
from foamsim.micromechanics import (
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
    particle_crush_onset,
)

app = typer.Typer(no_args_is_help=True); console = Console()


@app.command()
def estimate(matrix: str = "epoxy", grade: str = "K46", vf: float = 0.4):
    m = MATERIALS[matrix]; p = hollow_particle(grade)
    out = {"HP-MT": hollow_particle_mori_tanaka(m, p, vf).as_dict(), "HP-DS": hollow_particle_differential(m, p, vf).as_dict(),
           "HS_bounds": hashin_shtrikman_bounds(m, p, vf), "eta": p.eta, "crush_onset_mpa": particle_crush_onset(p, m, vf)}
    console.print_json(json.dumps(out))


@app.command()
def sweep(matrix: str = "epoxy", grade: str = "K46", out: str = "sweep.csv"):
    import numpy as np
    import pandas as pd
    m = MATERIALS[matrix]; p = hollow_particle(grade)
    rows = [hollow_particle_mori_tanaka(m, p, vf).as_dict() for vf in np.linspace(0, 0.6, 13)]
    pd.DataFrame(rows).to_csv(out, index=False); console.print(f"wrote {out}")


@app.command()
def fe(matrix: str = "epoxy", grade: str = "K46", vf: float = 0.3, n: int = 24, spheres: int = 20, mode: str = "equivalent"):
    from foamsim.fem import homogenize
    from foamsim.rve import random_packing
    m = MATERIALS[matrix]; p = hollow_particle(grade)
    rve = random_packing(vf, n_spheres=spheres, eta=p.eta)
    e = homogenize(rve, m, p, n=n, mode=mode)
    console.print_json(json.dumps({**e.as_dict(), "analytical_MT": hollow_particle_mori_tanaka(m, p, rve.vf).as_dict()}))


if __name__ == "__main__":
    app()
