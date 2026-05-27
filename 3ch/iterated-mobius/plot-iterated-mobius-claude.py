#!/usr/bin/env python3

#!/usr/bin/env python3
"""
Iterated Möbius Transform Visualizer
======================================
Plots families of circles under repeated application of a Möbius (linear
fractional) transform  f(z) = (az + b) / (cz + d).

Under any Möbius transform, circles and lines map to circles and lines.
Iterating produces the three classic orbit families:

  Hyperbolic   – two real fixed points; circles shrink toward one, expand from other
  Elliptic     – two conjugate fixed points; circles rotate around them
  Loxodromic   – complex fixed points; circles spiral (combination of both)

Usage
-----
  python mobius.py                     # renders the three preset examples
  python mobius.py --save out.png      # save instead of showing

Extending
---------
  1. Define a 2×2 complex numpy array M = [[a,b],[c,d]] (need not be normalised).
  2. Pick seed circles as (center: complex, radius: float) tuples.
  3. Call plot_iterated_mobius(M, seed_circles, ...) on a matplotlib Axes.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


# ── Core Möbius math ──────────────────────────────────────────────────────────

def mobius_inv(M: np.ndarray) -> np.ndarray:
    """Inverse of Möbius matrix [[a,b],[c,d]] = [[d,-b],[-c,a]] / det."""
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    det = a * d - b * c
    return np.array([[d, -b], [-c, a]], dtype=complex) / det


def _circle_from_three(z1: complex, z2: complex, z3: complex):
    """
    Circumcircle of three complex points.
    Returns (center: complex, radius: float) or (None, None) if collinear.
    """
    x1, y1 = z1.real, z1.imag
    x2, y2 = z2.real, z2.imag
    x3, y3 = z3.real, z3.imag

    D = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-14:
        return None, None  # collinear / maps to a line

    r1sq = x1 ** 2 + y1 ** 2
    r2sq = x2 ** 2 + y2 ** 2
    r3sq = x3 ** 2 + y3 ** 2

    ux = (r1sq * (y2 - y3) + r2sq * (y3 - y1) + r3sq * (y1 - y2)) / D
    uy = (r1sq * (x3 - x2) + r2sq * (x1 - x3) + r3sq * (x2 - x1)) / D
    center = complex(ux, uy)
    return center, abs(z1 - center)


def transform_circle(center: complex, radius: float, M: np.ndarray):
    """
    Image of circle (center, radius) under Möbius matrix M.
    Uses three representative points; avoids sampling near the pole -d/c.
    Returns (center, radius) or (None, None) if the image is a line.
    """
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]

    # Choose three angles spread around the circle, rotated 15° from canonical
    # to reduce chance of landing on the pole when c ≠ 0.
    base = np.pi / 12
    angles = np.array([base, base + 2 * np.pi / 3, base + 4 * np.pi / 3])
    pts = center + radius * np.exp(1j * angles)

    # If any sample is too close to the pole, perturb slightly
    pole = -d / c if abs(c) > 1e-15 else None
    if pole is not None:
        for i, p in enumerate(pts):
            if abs(p - pole) < radius * 0.05:
                angles[i] += np.pi / 7
                pts[i] = center + radius * np.exp(1j * angles[i])

    mapped = (a * pts + b) / (c * pts + d)

    # Sanity check
    if not np.all(np.isfinite(mapped)):
        return None, None

    return _circle_from_three(mapped[0], mapped[1], mapped[2])


# ── Transform constructors ────────────────────────────────────────────────────

def _conjugate(M_0inf: np.ndarray, p: complex, q: complex) -> np.ndarray:
    """
    Conjugate M_0inf (which fixes 0 and ∞) so the result fixes p and q instead.
    The conjugating map S(z) = (q·z + p)/(z + 1) sends 0 → p, ∞ → q.
    """
    S = np.array([[q, p], [1, 1]], dtype=complex)
    S_inv = mobius_inv(S)
    return S @ M_0inf @ S_inv


def hyperbolic(k: float = 2.5, p: complex = -1 + 0j, q: complex = 1 + 0j) -> np.ndarray:
    """
    Hyperbolic transform with multiplier k (real > 1) and fixed points p, q.
    Canonical form: f(z) = k·z   (fixes 0 and ∞).
    """
    assert k > 1, "k must be real and > 1 for hyperbolic"
    M0 = np.array([[k, 0], [0, 1]], dtype=complex)
    return _conjugate(M0, p, q)


def elliptic(theta: float = np.pi / 6, p: complex = -1j, q: complex = 1j) -> np.ndarray:
    """
    Elliptic transform with rotation angle θ and fixed points p, q.
    Canonical form: f(z) = e^(iθ)·z   (fixes 0 and ∞).
    """
    M0 = np.array([[np.exp(1j * theta), 0], [0, 1]], dtype=complex)
    return _conjugate(M0, p, q)


def loxodromic(k: float = 1.35, theta: float = np.pi / 7,
               p: complex = -0.5 - 0.5j, q: complex = 0.5 + 0.5j) -> np.ndarray:
    """
    Loxodromic transform: spiral (hyperbolic × elliptic).
    Canonical form: f(z) = k·e^(iθ)·z   (fixes 0 and ∞).
    """
    M0 = np.array([[k * np.exp(1j * theta), 0], [0, 1]], dtype=complex)
    return _conjugate(M0, p, q)


def classify(M: np.ndarray) -> str:
    """Classify a Möbius transform by its trace squared."""
    a, d = M[0, 0], M[1, 1]
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    tr_sq = (a + d) ** 2 / det        # normalise
    t = tr_sq.real
    i = abs(tr_sq.imag)
    if i > 1e-8:
        return "loxodromic"
    if abs(t - 4) < 1e-8:
        return "parabolic"
    if t > 4:
        return "hyperbolic"
    if 0 <= t < 4:
        return "elliptic"
    return "loxodromic"


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_circle(ax, center: complex, radius: float, **kw):
    patch = plt.Circle((center.real, center.imag), radius, fill=False, **kw)
    ax.add_patch(patch)


def plot_iterated_mobius(
    M: np.ndarray,
    seed_circles: list,
    n_iter: int = 20,
    ax=None,
    fwd_color: str = "royalblue",
    bwd_color: str = "gold",
    seed_color: str = "white",
    linewidth: float = 0.75,
    alpha_start: float = 1.0,
    alpha_min: float = 0.12,
    xlim: tuple = (-3, 3),
    ylim: tuple = (-3, 3),
    title: str = "",
) -> plt.Axes:
    """
    Plot iterated Möbius orbits of seed_circles on ax.

    Parameters
    ----------
    M            : 2×2 complex array [[a,b],[c,d]]
    seed_circles : list of (center: complex, radius: float)
    n_iter       : number of forward (and backward) iterations per seed
    ax           : matplotlib Axes; created if None
    fwd_color    : colour for forward iterates
    bwd_color    : colour for backward iterates (None → same as fwd)
    seed_color   : colour of the seed circle(s)
    xlim, ylim   : axis limits
    title        : subplot title
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")

    ax.set_facecolor("black")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, color="white", fontsize=13, pad=8)

    if bwd_color is None:
        bwd_color = fwd_color

    M_inv = mobius_inv(M)

    def _alpha(i: int) -> float:
        decay = (alpha_start - alpha_min) / n_iter
        return max(alpha_min, alpha_start - i * decay)

    for (c0, r0) in seed_circles:
        # ── forward orbit ──────────────────────────────────────────────
        fwd = []
        cc, rr = c0, r0
        for _ in range(n_iter):
            cc, rr = transform_circle(cc, rr, M)
            if cc is None or not np.isfinite(cc) or rr > 1e5 or rr < 1e-9:
                break
            fwd.append((cc, rr))

        # ── backward orbit ─────────────────────────────────────────────
        bwd = []
        cc, rr = c0, r0
        for _ in range(n_iter):
            cc, rr = transform_circle(cc, rr, M_inv)
            if cc is None or not np.isfinite(cc) or rr > 1e5 or rr < 1e-9:
                break
            bwd.append((cc, rr))

        # ── draw ───────────────────────────────────────────────────────
        _draw_circle(ax, c0, r0, color=seed_color, linewidth=linewidth * 1.2)

        for i, (cen, rad) in enumerate(fwd):
            _draw_circle(ax, cen, rad, color=fwd_color, linewidth=linewidth,
                         alpha=_alpha(i))

        for i, (cen, rad) in enumerate(bwd):
            _draw_circle(ax, cen, rad, color=bwd_color, linewidth=linewidth,
                         alpha=_alpha(i))

    return ax


# ── Preset gallery ────────────────────────────────────────────────────────────

def demo():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor="black")
    fig.patch.set_facecolor("black")
    fig.suptitle("Iterated Möbius Transforms", color="white", fontsize=15, y=1.01)

    # ── Hyperbolic ────────────────────────────────────────────────────────────
    Mh = hyperbolic(k=2.8, p=-1 + 0j, q=1 + 0j)
    seeds_h = [
        (0.3j,         0.25),
        (-0.3j,        0.25),
        (0 + 0j,       0.7),
        (0.6 + 0j,     0.15),
        (-0.6 + 0j,    0.15),
    ]
    plot_iterated_mobius(Mh, seeds_h, n_iter=3, ax=axes[0],
                         fwd_color="royalblue", bwd_color="royalblue",
                         xlim=(-2.8, 2.8), ylim=(-2.8, 2.8),
                         title="Hyperbolic  (k = 2.8)")

    # ── Elliptic ──────────────────────────────────────────────────────────────
    Me = elliptic(theta=np.pi / 7, p=-0.8j, q=0.8j)
    seeds_e = [
        (0.9 + 0j,     0.35),
        (1.6 + 0j,     0.3),
        (0.3 + 0.3j,   0.18),
        (-1.0 + 0j,    0.2),
    ]
    plot_iterated_mobius(Me, seeds_e, n_iter=2, ax=axes[1],
                         fwd_color="royalblue", bwd_color="gold",
                         xlim=(-2.8, 2.8), ylim=(-2.8, 2.8),
                         title=f"Elliptic  (θ = π/7 ≈ {np.pi/7:.3f})")

    # ── Loxodromic ────────────────────────────────────────────────────────────
    Ml = loxodromic(k=1.35, theta=np.pi / 7,
                    p=-0.5 - 0.5j, q=0.8 + 0.6j)
    seeds_l = [
        (0.2 + 0.2j,   0.25),
        (-0.5 + 0.8j,  0.2),
        (1.0 - 0.3j,   0.3),
    ]
    plot_iterated_mobius(Ml, seeds_l, n_iter=1, ax=axes[2],
                         fwd_color="royalblue", bwd_color="gold",
                         xlim=(-3.2, 3.2), ylim=(-3.2, 3.2),
                         title="Loxodromic  (k = 1.35, θ = π/7)")

    plt.tight_layout()
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterated Möbius transform plotter")
    parser.add_argument("--save", metavar="PATH", default=None,
                        help="Save to file instead of displaying (e.g. out.png)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    fig = demo()

    if args.save:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight", facecolor="black")
        print(f"Saved to {args.save}")
    else:
        plt.show()
