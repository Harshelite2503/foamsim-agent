"""Representative volume elements: random sphere packing + voxelisation.

Random sequential adsorption (RSA) of equal spheres in a periodic unit cube. RSA saturates
near vf ~ 0.38 for monodisperse spheres; above that we use a short shaking (Lubachevsky-Stillinger-
like) relaxation to reach up to ~0.55. Higher fractions need polydispersity - documented limit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RVE:
    centers: np.ndarray      # (n,3) in [0,1)^3, periodic
    radius: float            # sphere outer radius in box units
    eta: float               # inner/outer radius ratio
    target_vf: float

    @property
    def vf(self) -> float:
        return len(self.centers) * 4 / 3 * np.pi * self.radius ** 3

    def voxelize(self, n: int) -> np.ndarray:
        """Phase id per voxel: 0 matrix, 1 shell, 2 void core. Shape (n,n,n)."""
        g = (np.arange(n) + 0.5) / n
        X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
        P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        dmin = np.full(len(P), np.inf)
        for c in self.centers:
            d = P - c; d -= np.round(d)  # minimum image
            dmin = np.minimum(dmin, np.linalg.norm(d, axis=1))
        phase = np.zeros(len(P), dtype=np.int8)
        phase[dmin < self.radius] = 1
        phase[dmin < self.eta * self.radius] = 2
        return phase.reshape(n, n, n)

    def shell_thickness_voxels(self, n: int) -> float:
        return (1 - self.eta) * self.radius * n


def _min_image_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b; d -= np.round(d)
    return np.linalg.norm(d, axis=-1)


def random_packing(vf: float, n_spheres: int = 30, eta: float = 0.9, seed: int = 0,
                   max_tries: int = 200_000) -> RVE:
    """Periodic RSA packing of n_spheres equal spheres to reach volume fraction vf.
    Raises if vf is not reachable (RSA jamming ~0.38 without relaxation, ~0.55 with)."""
    if not 0 < vf < 0.64:
        raise ValueError("vf must be in (0, 0.64) - random close packing limit")
    rng = np.random.default_rng(seed)
    r = (3 * vf / (4 * np.pi * n_spheres)) ** (1 / 3)
    if 2 * r > 0.5:
        raise ValueError("spheres too large for the periodic box; increase n_spheres")
    centers = []
    tries = 0
    while len(centers) < n_spheres and tries < max_tries:
        c = rng.random(3); tries += 1
        if all(_min_image_dist(c, x) >= 2 * r for x in centers):
            centers.append(c)
    centers = np.array(centers)
    if len(centers) < n_spheres:  # relaxation: shrink radius, place all, then push apart
        centers = _relax(rng, n_spheres, r)
    return RVE(centers, r, eta, vf)


def _relax(rng, n, r, steps: int = 4000) -> np.ndarray:
    c = rng.random((n, 3)); r_now = 0.5 * r
    for step in range(steps):
        r_now = min(r, r_now * 1.002)
        for i in range(n):
            d = c[i] - c; d -= np.round(d); dist = np.linalg.norm(d, axis=1); dist[i] = np.inf
            over = dist < 2 * r_now
            if over.any():
                push = (d[over] / dist[over, None]) * (2 * r_now - dist[over, None]) * 0.5
                c[i] = (c[i] + push.sum(0)) % 1.0
        if r_now >= r:
            d = c[:, None] - c[None]; d -= np.round(d); dist = np.linalg.norm(d, axis=2); np.fill_diagonal(dist, np.inf)
            if dist.min() >= 2 * r * 0.999:
                return c
    raise RuntimeError(f"could not pack {n} spheres at radius {r:.3f}; reduce vf or n_spheres")
