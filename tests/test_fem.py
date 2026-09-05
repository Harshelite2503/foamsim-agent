import numpy as np
import pytest

from foamsim import MATERIALS, hollow_particle
from foamsim.fem import ResolutionError, homogenize, homogenize_homogeneous
from foamsim.micromechanics import hashin_shtrikman_bounds
from foamsim.rve import random_packing

EP = MATERIALS["epoxy"]


def test_homogeneous_box_returns_matrix():
    e = homogenize_homogeneous(EP, n=4)
    assert abs(e.E - EP.E) / EP.E < 1e-6 and abs(e.nu - EP.nu) < 1e-6


def test_equivalent_mode_within_hs_bounds():
    p = hollow_particle("K46"); rve = random_packing(0.3, n_spheres=12, eta=p.eta, seed=1)
    e = homogenize(rve, EP, p, n=16, mode="equivalent")
    b = hashin_shtrikman_bounds(EP, p, rve.vf)
    assert 0.8 * b["E_lo"] < e.E < 1.2 * b["E_hi"], (e.E, b)


def test_shell_mode_resolution_guard():
    p = hollow_particle("K46"); rve = random_packing(0.3, n_spheres=12, eta=p.eta, seed=1)
    with pytest.raises(ResolutionError):
        homogenize(rve, EP, p, n=16, mode="shell")


def test_packing_reaches_vf():
    rve = random_packing(0.45, n_spheres=20, seed=2)
    assert abs(rve.vf - 0.45) < 0.01
    d = rve.centers[:, None] - rve.centers[None]; d -= np.round(d)
    dist = np.linalg.norm(d, axis=2); np.fill_diagonal(dist, np.inf)
    assert dist.min() >= 2 * rve.radius * 0.99
