"""Constituent materials and hollow-particle geometry.

Units: moduli in MPa, density in g/cm^3, lengths in micrometres. Poisson ratio dimensionless.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Isotropic:
    """Isotropic linear-elastic solid."""
    name: str
    E: float          # Young's modulus, MPa
    nu: float         # Poisson ratio
    rho: float        # density, g/cm^3

    @property
    def K(self) -> float:
        return self.E / (3 * (1 - 2 * self.nu))

    @property
    def G(self) -> float:
        return self.E / (2 * (1 + self.nu))

    @staticmethod
    def from_KG(name: str, K: float, G: float, rho: float) -> Isotropic:
        E = 9 * K * G / (3 * K + G); nu = (3 * K - 2 * G) / (2 * (3 * K + G))
        return Isotropic(name, E, nu, rho)


# Representative constituent values (literature-typical; override for specific grades).
MATERIALS: dict[str, Isotropic] = {
    "epoxy":        Isotropic("epoxy", 3000.0, 0.35, 1.18),
    "vinyl_ester":  Isotropic("vinyl_ester", 3200.0, 0.35, 1.13),
    "polyurethane": Isotropic("polyurethane", 1500.0, 0.40, 1.15),
    "hdpe":         Isotropic("hdpe", 1000.0, 0.42, 0.95),
    "pdms":         Isotropic("pdms", 2.5, 0.49, 1.03),
    "aluminum":     Isotropic("aluminum", 70000.0, 0.33, 2.70),
    "alsi12":       Isotropic("alsi12", 75000.0, 0.33, 2.65),
    "magnesium":    Isotropic("magnesium", 45000.0, 0.35, 1.74),
    "glass":        Isotropic("glass", 60000.0, 0.21, 2.54),   # borosilicate microballoon wall
    "alumina":      Isotropic("alumina", 300000.0, 0.22, 3.90),
    "silica_shell": Isotropic("silica_shell", 70000.0, 0.17, 2.20),
}


@dataclass(frozen=True)
class HollowParticle:
    """Thin-walled hollow sphere: shell material, radius ratio eta = r_inner / r_outer."""
    shell: Isotropic
    eta: float                      # 0 = solid sphere, ->1 = vanishing wall
    diameter_um: float = 40.0

    def __post_init__(self):
        if not 0.0 <= self.eta < 1.0:
            raise ValueError(f"eta must be in [0,1), got {self.eta}")

    @property
    def wall_volume_fraction(self) -> float:
        return 1.0 - self.eta ** 3

    @property
    def true_density(self) -> float:
        """Particle (true) density = shell density x wall volume fraction."""
        return self.shell.rho * self.wall_volume_fraction

    @staticmethod
    def from_true_density(shell: Isotropic, true_density: float, diameter_um: float = 40.0) -> HollowParticle:
        """Infer eta from the manufacturer's true density (e.g. 3M K46: 0.46 g/cm^3)."""
        if not 0 < true_density <= shell.rho:
            raise ValueError("true_density must be in (0, shell density]")
        eta = (1.0 - true_density / shell.rho) ** (1.0 / 3.0)
        return HollowParticle(shell, eta, diameter_um)


def hollow_particle(grade: str) -> HollowParticle:
    """Common 3M glass-bubble grades by true density."""
    table = {"K1": 0.125, "K15": 0.15, "K20": 0.20, "K25": 0.25, "S22": 0.22, "S32": 0.32, "S38": 0.38,
             "K46": 0.46, "S60": 0.60, "iM16K": 0.46, "iM30K": 0.60, "H50": 0.50}
    if grade not in table:
        raise KeyError(f"unknown grade {grade}; known: {sorted(table)}")
    return HollowParticle.from_true_density(MATERIALS["glass"], table[grade])
