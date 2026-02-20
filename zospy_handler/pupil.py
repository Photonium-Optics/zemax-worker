"""Shared pupil coordinate generation functions.

Used by both ray_tracing.py (diagnostic) and ray_analysis.py (unified).
"""

import numpy as np


def generate_hexapolar_coords(num_rays: int) -> list[tuple[float, float]]:
    """Generate hexapolar (concentric ring) pupil coordinates.

    Hexapolar layout:
      Ring 0: 1 ray at center
      Ring k: 6*k rays equally spaced at radius k/num_rings
    Total rays for N rings: 1 + 3*N*(N+1)

    Args:
        num_rays: Approximate target number of rays. The actual count will be
                  the nearest hexapolar total (1, 7, 19, 37, 61, 91, ...).
    """
    # Solve 1 + 3*N*(N+1) >= num_rays for N
    num_rings = 0
    while 1 + 3 * num_rings * (num_rings + 1) < num_rays:
        num_rings += 1

    coords: list[tuple[float, float]] = [(0.0, 0.0)]
    for ring in range(1, num_rings + 1):
        r = ring / num_rings  # normalized radius [0, 1]
        n_pts = 6 * ring
        for j in range(n_pts):
            theta = 2.0 * np.pi * j / n_pts
            coords.append((r * np.cos(theta), r * np.sin(theta)))
    return coords


def generate_square_grid_coords(num_rays: int) -> list[tuple[float, float]]:
    """Generate square grid pupil coordinates clipped to unit circle."""
    grid_size = int(np.sqrt(num_rays))
    coords: list[tuple[float, float]] = []
    for px in np.linspace(-1, 1, grid_size):
        for py in np.linspace(-1, 1, grid_size):
            if px**2 + py**2 <= 1:
                coords.append((float(px), float(py)))
    return coords


def generate_random_coords(num_rays: int, seed: int = 42) -> list[tuple[float, float]]:
    """Generate random pupil coordinates within the unit circle.

    Uses rejection sampling with a deterministic seed for reproducibility.

    Args:
        num_rays: Exact number of rays to generate.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed=seed)
    coords: list[tuple[float, float]] = []
    while len(coords) < num_rays:
        batch = max(num_rays - len(coords), 64)
        px = rng.uniform(-1, 1, batch)
        py = rng.uniform(-1, 1, batch)
        mask = px**2 + py**2 <= 1.0
        for p, q in zip(px[mask], py[mask]):
            if len(coords) >= num_rays:
                break
            coords.append((float(p), float(q)))
    return coords
