"""Analytical homogenization for hollow-particle composites.

All models take an isotropic matrix, a HollowParticle, and a particle volume fraction vf
(fraction of composite volume occupied by particles, INCLUDING their hollow cores).

Models
------
density                      rule of mixtures with particle true density (+ optional matrix porosity)
hs_estimate                  general Hashin-Shtrikman (Walpole) estimate for n isotropic phases
                             against a reference medium; gives HS bounds and the exact hollow-sphere
                             bulk modulus as special cases
hollow_sphere_equivalent     hollow sphere -> equivalent homogeneous solid particle (K_p, G_p):
                             K_p exact (Hashin 1962 composite spheres, void core),
                             G_p = HS upper bound of the porous shell (no exact result exists)
mori_tanaka_spheres          Mori-Tanaka (Benveniste 1987) for isotropic spheres in a matrix
hollow_particle_mori_tanaka  equivalent particle -> Mori-Tanaka                     (HP-MT)
hollow_particle_differential equivalent particle -> differential scheme ODE         (HP-DS)
hashin_shtrikman_bounds      HS bounds for matrix + equivalent particle
gibson_ashby                 cellular-solid scaling E = C E_s (rho/rho_s)^n
particle_crush_onset         order-of-magnitude compressive stress at microballoon buckling

References: Hashin (1962) J. Appl. Mech.; Hashin & Shtrikman (1963) JMPS; Benveniste (1987)
Mech. Mater.; McLaughlin (1977) IJES; Porfiri & Gupta (2009) Compos. B; Bardella & Genna (2001)
IJSS; Gibson & Ashby (1997).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from foamsim.materials import HollowParticle, Isotropic

RCP = 0.64  # random close packing of monodisperse spheres


@dataclass(frozen=True)
class Effective:
    K: float
    G: float
    rho: float
    vf: float
    model: str

    @property
    def E(self) -> float:
        return 9 * self.K * self.G / (3 * self.K + self.G)

    @property
    def nu(self) -> float:
        return (3 * self.K - 2 * self.G) / (2 * (3 * self.K + self.G))

    def as_dict(self) -> dict:
        return {"model": self.model, "vf": self.vf, "E_mpa": self.E, "K_mpa": self.K, "G_mpa": self.G,
                "nu": self.nu, "rho_g_cc": self.rho, "specific_E": self.E / self.rho}


def _check_vf(vf: float) -> None:
    if not 0.0 <= vf <= 1.0:
        raise ValueError(f"vf must be in [0,1], got {vf}")
    if vf > RCP:
        raise ValueError(f"vf={vf} exceeds random close packing ({RCP}) of monodisperse spheres")


def density(matrix: Isotropic, particle: HollowParticle, vf: float, matrix_porosity: float = 0.0) -> float:
    """Composite density, g/cm^3. matrix_porosity = void fraction of the matrix phase."""
    _check_vf(vf)
    return vf * particle.true_density + (1 - vf) * (1 - matrix_porosity) * matrix.rho


def hs_estimate(K0: float, G0: float, phases: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Hashin-Shtrikman/Walpole estimate against reference medium (K0, G0).
    phases = [(K_i, G_i, c_i), ...] with sum c_i = 1.
        K* = [sum c_i/(K_i + 4G0/3)]^-1 - 4G0/3
        G* = [sum c_i/(G_i + z0)]^-1 - z0,   z0 = G0/6 (9K0 + 8G0)/(K0 + 2G0)
    Reference = stiffest phase -> upper bound; softest -> lower bound."""
    a = 4 * G0 / 3
    z = G0 / 6 * (9 * K0 + 8 * G0) / (K0 + 2 * G0)
    K = 1.0 / sum(c / (Ki + a) for Ki, _, c in phases) - a
    G = 1.0 / sum(c / (Gi + z) for _, Gi, c in phases) - z
    return K, G


def hollow_sphere_equivalent(particle: HollowParticle) -> Isotropic:
    """Equivalent homogeneous solid sphere for a hollow sphere with a void core (f = eta^3).
    K_p = 4 G_s K_s (1-f) / (3 K_s f + 4 G_s)  (exact, Hashin composite-sphere assemblage);
    G_p = HS upper bound of shell with void fraction f (used by Porfiri & Gupta-type schemes)."""
    s = particle.shell; f = particle.eta ** 3
    K_p, G_p = hs_estimate(s.K, s.G, [(s.K, s.G, 1 - f), (0.0, 0.0, f)])
    K_exact = 4 * s.G * s.K * (1 - f) / (3 * s.K * f + 4 * s.G)
    assert abs(K_p - K_exact) < 1e-6 * max(1.0, K_exact), "HS void estimate must reproduce Hashin K"
    return Isotropic.from_KG(f"eq({s.name},eta={particle.eta:.3f})", K_p, G_p, particle.true_density)


def mori_tanaka_spheres(matrix: Isotropic, inclusion: Isotropic, vf: float) -> tuple[float, float]:
    """Mori-Tanaka (Benveniste 1987) effective K, G for isotropic spherical inclusions.
    Identical to the HS estimate with the matrix as reference medium."""
    return hs_estimate(matrix.K, matrix.G, [(matrix.K, matrix.G, 1 - vf), (inclusion.K, inclusion.G, vf)])


def _porous_matrix(matrix: Isotropic, porosity: float) -> Isotropic:
    if porosity <= 0:
        return matrix
    K, G = hs_estimate(matrix.K, matrix.G, [(matrix.K, matrix.G, 1 - porosity), (0.0, 0.0, porosity)])
    return Isotropic.from_KG(f"{matrix.name}(porosity={porosity})", K, G, matrix.rho * (1 - porosity))


def hollow_particle_mori_tanaka(matrix: Isotropic, particle: HollowParticle, vf: float,
                                matrix_porosity: float = 0.0) -> Effective:
    """HP-MT: equivalent-particle Mori-Tanaka estimate."""
    _check_vf(vf)
    p = hollow_sphere_equivalent(particle); m = _porous_matrix(matrix, matrix_porosity)
    K, G = mori_tanaka_spheres(m, p, vf)
    return Effective(K, G, density(matrix, particle, vf, matrix_porosity), vf, "HP-MT")


def hollow_particle_differential(matrix: Isotropic, particle: HollowParticle, vf: float,
                                 matrix_porosity: float = 0.0) -> Effective:
    """HP-DS: differential scheme (McLaughlin 1977) with the equivalent particle:
        dK/dphi = (K_p - K)/(1-phi) * (K + 4G/3)/(K_p + 4G/3)
        dG/dphi = (G_p - G)/(1-phi) * (G + z)/(G_p + z),  z = G/6 (9K+8G)/(K+2G)."""
    _check_vf(vf)
    p = hollow_sphere_equivalent(particle); m = _porous_matrix(matrix, matrix_porosity)

    def rhs(phi, y):
        K, G = y
        z = G / 6 * (9 * K + 8 * G) / (K + 2 * G)
        return [(p.K - K) / (1 - phi) * (K + 4 * G / 3) / (p.K + 4 * G / 3),
                (p.G - G) / (1 - phi) * (G + z) / (p.G + z)]

    if vf == 0:
        K, G = m.K, m.G
    else:
        sol = solve_ivp(rhs, (0, vf), [m.K, m.G], rtol=1e-9, atol=1e-12)
        K, G = float(sol.y[0, -1]), float(sol.y[1, -1])
    return Effective(K, G, density(matrix, particle, vf, matrix_porosity), vf, "HP-DS")


def hashin_shtrikman_bounds(matrix: Isotropic, particle: HollowParticle, vf: float) -> dict:
    """HS lower/upper bounds for matrix + equivalent particle (two isotropic phases)."""
    _check_vf(vf)
    p = hollow_sphere_equivalent(particle)
    phases = [(matrix.K, matrix.G, 1 - vf), (p.K, p.G, vf)]
    soft, stiff = (matrix, p) if matrix.G <= p.G else (p, matrix)
    K_lo, G_lo = hs_estimate(soft.K, soft.G, phases); K_hi, G_hi = hs_estimate(stiff.K, stiff.G, phases)
    E = lambda K, G: 9 * K * G / (3 * K + G)
    return {"K_lo": K_lo, "K_hi": K_hi, "G_lo": G_lo, "G_hi": G_hi, "E_lo": E(K_lo, G_lo), "E_hi": E(K_hi, G_hi)}


def gibson_ashby(matrix: Isotropic, relative_density: float, C: float = 1.0, n: float = 2.0) -> float:
    """Gibson-Ashby modulus scaling for cellular solids: E = C E_s (rho/rho_s)^n, MPa."""
    if not 0 < relative_density <= 1:
        raise ValueError("relative_density in (0,1]")
    return C * matrix.E * relative_density ** n


def particle_crush_onset(particle: HollowParticle, matrix: Isotropic, vf: float) -> float:
    """Order-of-magnitude macroscopic compressive stress (MPa) at microballoon buckling.
    Zoelly shell buckling: p_cr = 2 E_s (t/R)^2 / sqrt(3(1-nu_s^2)), t/R = 1 - eta.
    Particle mean stress under macroscopic mean stress p (dilute Eshelby estimate):
        p_particle = p * K_p (3K_m + 4G_m) / (K_m (3K_p + 4G_m)).
    Uniaxial stress sigma gives p = sigma/3  ->  sigma_crush = 3 p_cr / conc."""
    _check_vf(vf)
    s = particle.shell; p_cr = 2 * s.E * (1 - particle.eta) ** 2 / np.sqrt(3 * (1 - s.nu ** 2))
    p = hollow_sphere_equivalent(particle)
    conc = p.K * (3 * matrix.K + 4 * matrix.G) / (matrix.K * (3 * p.K + 4 * matrix.G))
    return float(3 * p_cr / conc)
