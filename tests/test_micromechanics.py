import numpy as np
import pytest

from foamsim import MATERIALS, hollow_particle
from foamsim.materials import HollowParticle
from foamsim.micromechanics import (
    density,
    gibson_ashby,
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
    hollow_sphere_equivalent,
    mori_tanaka_spheres,
)

EP = MATERIALS["epoxy"]; GL = MATERIALS["glass"]


def test_density_rule_of_mixtures():
    k46 = hollow_particle("K46")
    assert abs(k46.true_density - 0.46) < 1e-9
    assert abs(density(EP, k46, 0.4) - (0.4 * 0.46 + 0.6 * 1.18)) < 1e-9


def test_vf_zero_returns_matrix():
    for fn in (hollow_particle_mori_tanaka, hollow_particle_differential):
        e = fn(EP, hollow_particle("K46"), 0.0)
        assert abs(e.E - EP.E) < 1e-6 and abs(e.nu - EP.nu) < 1e-9


def test_solid_sphere_limit_matches_mori_tanaka():
    solid = HollowParticle(GL, 0.0)
    K, G = mori_tanaka_spheres(EP, GL, 0.3)
    e = hollow_particle_mori_tanaka(EP, solid, 0.3)
    assert abs(e.K - K) < 1e-6 and abs(e.G - G) < 1e-6


def test_hollow_sphere_equivalent_limits():
    assert abs(hollow_sphere_equivalent(HollowParticle(GL, 0.0)).E - GL.E) < 1e-6
    thin = hollow_sphere_equivalent(HollowParticle(GL, 0.999))
    assert thin.E < 0.01 * GL.E
    # monotone decreasing with eta
    Es = [hollow_sphere_equivalent(HollowParticle(GL, eta)).E for eta in np.linspace(0, 0.99, 20)]
    assert all(a > b for a, b in zip(Es, Es[1:]))


def test_estimates_within_hs_bounds():
    for grade in ("K1", "S38", "K46", "S60"):
        p = hollow_particle(grade)
        for vf in (0.1, 0.3, 0.5, 0.6):
            b = hashin_shtrikman_bounds(EP, p, vf)
            for fn in (hollow_particle_mori_tanaka, hollow_particle_differential):
                e = fn(EP, p, vf)
                assert b["E_lo"] - 1e-6 <= e.E <= b["E_hi"] + 1e-6, (grade, vf, fn.__name__)


def test_stiff_particles_raise_modulus_light_particles_lower_it():
    assert hollow_particle_mori_tanaka(EP, hollow_particle("S60"), 0.4).E > EP.E
    assert hollow_particle_mori_tanaka(EP, hollow_particle("K1"), 0.4).E < EP.E


def test_vf_above_packing_rejected():
    with pytest.raises(ValueError):
        hollow_particle_mori_tanaka(EP, hollow_particle("K46"), 0.7)
    with pytest.raises(ValueError):
        HollowParticle(GL, 1.0)


def test_gibson_ashby_scaling():
    assert abs(gibson_ashby(MATERIALS["aluminum"], 0.5) - 0.25 * 70000) < 1e-6


def test_literature_sanity_epoxy_k46_40pct():
    """Gupta et al. report E ~ 2.0-2.6 GPa for epoxy/K46 at ~30-40 vol%; MT and DS should land in 1.5-4 GPa."""
    e = hollow_particle_mori_tanaka(EP, hollow_particle("K46"), 0.4)
    assert 1500 < e.E < 4000, e.E
