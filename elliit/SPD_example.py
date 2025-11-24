#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from pymanopt.manifolds import SymmetricPositiveDefinite
from scipy.linalg import sqrtm

# ---------- plotting helpers ----------


def setup_axes():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("SPD(2) manifold (visualized as ellipses)")
    ax.set_aspect("equal", adjustable="box")
    lim = 3.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    return fig, ax


def _ellipse_vertices(Sigma, scale=1.0, center=None, samples=60):
    """Return a set of 2D vertices describing the ellipse of Sigma."""
    assert Sigma.shape == (2, 2)
    if center is None:
        center = np.zeros(2)

    evals, evecs = np.linalg.eigh(Sigma)
    D = np.diag(evals)
    R = np.real(evecs @ np.sqrt(D + 0j))

    al = np.linspace(-np.pi, np.pi, samples)
    circle = np.array([np.cos(al), np.sin(al)])
    vertices = (scale * (R @ circle)).T + center
    return vertices, center


def plot_spd_ellipse(ax, P, *, scale=1.0, edgecolor="C0", linestyle="-", label=None, alpha=1.0):
    """Single helper that takes an SPD(2) matrix and draws its ellipse."""
    vertices, center = _ellipse_vertices(P, scale=scale)
    e = Polygon(
        vertices,
        closed=True,
        edgecolor=edgecolor,
        facecolor="none",
        linestyle=linestyle,
        linewidth=2,
        alpha=alpha,
    )
    ax.add_patch(e)
    if label is not None:
        # Pick the vertex furthest from the center for the label.
        # Nudge it slightly outward.
        label_pos_idx = np.argmax(np.linalg.norm(vertices, axis=1))
        label_pos = vertices[label_pos_idx]

        ax.text(
            label_pos[0] * 1.15,
            label_pos[1] * 1.15,
            label,
            color=edgecolor,
            ha="center",
            va="center",
        )


# ---------- manifold math + calls to plotting helpers ----------


def parallel_transport_spd(X, Y, V):
    """
    Parallel transport of a tangent vector V from X to Y on the SPD manifold.

    Equation:
        Gamma_{X->Y}(V) = A @ V @ A.T,
        with A = (Y @ X^(-1))^(1/2) = Y^(1/2) @ X^(-1/2)
    """
    x_sqrt = sqrtm(X)
    x_inv_sqrt = np.linalg.inv(x_sqrt)
    y_sqrt = sqrtm(Y)

    A = y_sqrt @ x_inv_sqrt
    transported_V = A @ V @ A.T

    return transported_V


def main():
    # SPD(2) manifold with affine-invariant metric
    manifold = SymmetricPositiveDefinite(2)

    # Base SPD matrix X
    X = manifold.random_point()

    # Tangent vectors at X
    U = manifold.random_tangent_vector(X)  # direction for exp/log
    W = manifold.random_tangent_vector(X)  # direction to be "transported"

    # Move along geodesic from X in direction U
    Y = manifold.exp(X, U)

    # Recover tangent at X pointing to Y
    V = manifold.log(X, Y)

    # Vector transport W from T_X to T_Y
    W_Y = parallel_transport_spd(X, Y, W)

    # Sanity checks
    print("‖V‖_X:", manifold.norm(X, V))
    print("dist(X, Y):", manifold.dist(X, Y))
    print("‖V‖_X - dist(X, Y):", manifold.norm(X, V) - manifold.dist(X, Y))

    # Plotting
    fig, ax = setup_axes()

    plot_spd_ellipse(ax, X, scale=1.0, edgecolor="C0", linestyle="-", label="X")
    plot_spd_ellipse(ax, Y, scale=1.0, edgecolor="C1", linestyle="-", label="Y = Exp_X(t*U)")

    # Use small steps along W and W_Y to generate nearby SPD matrices
    epsilon = 0.3
    Z_X = manifold.exp(X, epsilon * W)
    plot_spd_ellipse(ax, Z_X, scale=1.0, edgecolor="C2", linestyle="--", label="exp_X(eps * W)")
    Z_Y = manifold.exp(Y, epsilon * W_Y)
    plot_spd_ellipse(ax, Z_Y, scale=1.0, edgecolor="C3", linestyle="--", label="exp_Y(eps * W_Y)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
