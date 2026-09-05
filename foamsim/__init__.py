"""FoamSim: micromechanics simulation toolkit for hollow-particle (syntactic foam) composites."""
from foamsim.materials import MATERIALS, Isotropic, hollow_particle  # noqa: F401
from foamsim.micromechanics import (  # noqa: F401
    density,
    gibson_ashby,
    hashin_shtrikman_bounds,
    hollow_particle_differential,
    hollow_particle_mori_tanaka,
    hollow_sphere_equivalent,
    mori_tanaka_spheres,
)

__version__ = "0.1.0"
