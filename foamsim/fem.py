"""Voxel finite-element homogenization (scikit-fem, trilinear hexahedra, KUBC).

    eff = homogenize(rve, matrix, particle, n=24, mode="equivalent")

mode="equivalent": each hollow sphere is replaced by its equivalent homogeneous solid particle
                   (see micromechanics.hollow_sphere_equivalent). Resolves the particle-matrix
                   interaction numerically; cheap.
mode="shell":      resolves the glass shell and void core explicitly. Requires the shell to span
                   >= min_shell_voxels voxels or a ResolutionError is raised - thin microballoon
                   walls (eta > 0.9) need n >= ~64, which is slow on CPU.

Kinematic uniform boundary conditions (KUBC) give an upper-bound-type estimate for finite RVEs;
periodic BCs are more accurate but not implemented here. Six load cases -> Voigt C_eff, projected
onto the closest isotropic tensor.
"""
from __future__ import annotations

import numpy as np
import skfem
from skfem import (
    Basis,
    BilinearForm,
    ElementHex0,
    ElementHex1,
    ElementVector,
    MeshHex,
    condense,
    solve,
)
from skfem.helpers import ddot, sym_grad, trace

from foamsim.materials import HollowParticle, Isotropic
from foamsim.micromechanics import Effective, density, hollow_sphere_equivalent
from foamsim.rve import RVE

VOID_FACTOR = 1e-4  # void stiffness relative to matrix (keeps K non-singular)


class ResolutionError(ValueError):
    pass


@BilinearForm
def _elasticity(u, v, w):
    return w["lam"] * trace(sym_grad(u)) * trace(sym_grad(v)) + 2.0 * w["mu"] * ddot(sym_grad(u), sym_grad(v))


def _lame(m: Isotropic) -> tuple[float, float]:
    lam = m.E * m.nu / ((1 + m.nu) * (1 - 2 * m.nu)); mu = m.E / (2 * (1 + m.nu))
    return lam, mu


def homogenize(rve: RVE, matrix: Isotropic, particle: HollowParticle, n: int = 24, mode: str = "equivalent",
               min_shell_voxels: float = 2.0) -> Effective:
    phase = rve.voxelize(n)
    if mode == "shell" and rve.shell_thickness_voxels(n) < min_shell_voxels:
        raise ResolutionError(f"shell is {rve.shell_thickness_voxels(n):.2f} voxels thick (< {min_shell_voxels}); "
                              f"increase n to >= {int(np.ceil(min_shell_voxels / ((1 - rve.eta) * rve.radius)))}")
    lam_m, mu_m = _lame(matrix)
    if mode == "equivalent":
        eq = hollow_sphere_equivalent(particle); lam_p, mu_p = _lame(eq)
        lam = np.where(phase > 0, lam_p, lam_m); mu = np.where(phase > 0, mu_p, mu_m)
    elif mode == "shell":
        lam_s, mu_s = _lame(particle.shell)
        lam = np.select([phase == 1, phase == 2], [lam_s, lam_m * VOID_FACTOR], lam_m)
        mu = np.select([phase == 1, phase == 2], [mu_s, mu_m * VOID_FACTOR], mu_m)
    else:
        raise ValueError("mode must be 'equivalent' or 'shell'")

    mesh = MeshHex.init_tensor(*[np.linspace(0, 1, n + 1)] * 3)
    basis = Basis(mesh, ElementVector(ElementHex1()))
    basis0 = basis.with_element(ElementHex0())
    # map voxel (i,j,k) -> element index: init_tensor orders elements consistently with element centroids
    cent = mesh.p[:, mesh.t].mean(axis=1)  # (3, n_el)
    idx = np.minimum((cent * n).astype(int), n - 1)
    lam_e = lam[idx[0], idx[1], idx[2]]; mu_e = mu[idx[0], idx[1], idx[2]]
    K = _elasticity.assemble(basis, lam=basis0.interpolate(lam_e), mu=basis0.interpolate(mu_e))

    bdofs = basis.get_dofs().flatten()  # all boundary dof indices (nodal, interleaved u1,u2,u3)
    C = np.zeros((6, 6))
    cases = [np.array(e) for e in ([1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1])]
    def voigt_to_mat(e):
        return np.array([[e[0], e[5] / 2, e[4] / 2], [e[5] / 2, e[1], e[3] / 2], [e[4] / 2, e[3] / 2, e[2]]])
    for j, e in enumerate(cases):
        E = voigt_to_mat(e * 1e-3)
        u = basis.zeros()
        u[bdofs] = _kubc(basis, bdofs, E)
        u = solve(*condense(K, x=u, D=bdofs))
        sig = _mean_stress(basis, basis0, u, lam_e, mu_e)
        C[:, j] = sig / 1e-3
    C = 0.5 * (C + C.T)
    Kb = C[:3, :3].sum() / 9
    G = ((C[0, 0] + C[1, 1] + C[2, 2]) - (C[0, 1] + C[0, 2] + C[1, 2]) + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15
    return Effective(float(Kb), float(G), density(matrix, particle, rve.vf), rve.vf, f"FE-KUBC-{mode}-n{n}")


def _kubc(basis, bdofs, E):
    """u_i = E_ij x_j on boundary dofs; basis.doflocs is (3, ndofs); vector dofs interleave components."""
    loc = basis.doflocs[:, bdofs]                # (3, nb)
    comp = bdofs % 3                             # component of each vector dof (ElementVector ordering)
    return np.einsum("ij,jn->in", E, loc)[comp, np.arange(len(bdofs))]


def _mean_stress(basis, basis0, u, lam_e, mu_e):
    uh = basis.interpolate(u)
    eps = 0.5 * (uh.grad + np.transpose(uh.grad, (1, 0, 2, 3)))   # (3,3,nel,nqp)
    lam_q = basis0.interpolate(lam_e).value; mu_q = basis0.interpolate(mu_e).value
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    sig = 2 * mu_q * eps
    for i in range(3):
        sig[i, i] += lam_q * tr
    w = basis.dx  # quadrature weights (nel, nqp)
    vol = w.sum()
    avg = lambda a: float((a * w).sum() / vol)
    return np.array([avg(sig[0, 0]), avg(sig[1, 1]), avg(sig[2, 2]), avg(sig[1, 2]), avg(sig[0, 2]), avg(sig[0, 1])])


def homogenize_homogeneous(matrix: Isotropic, n: int = 6) -> Effective:
    """Sanity check: a homogeneous box must return the matrix moduli (KUBC is exact there)."""
    rve = RVE(np.zeros((0, 3)), 0.1, 0.9, 0.0)
    return homogenize(rve, matrix, HollowParticle(matrix, 0.0), n=n)


__all__ = ["ResolutionError", "homogenize", "homogenize_homogeneous", "skfem"]
