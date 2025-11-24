#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from pymanopt.manifolds import Sphere

# ---------- plotting helpers ----------


def setup_axes_3d():
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])
    lim = 1.5
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("S² manifold")
    return fig, ax


def plot_sphere(ax, *, n_theta=30, n_phi=30):
    phi, theta = np.mgrid[0 : np.pi : n_phi * 1j, 0 : 2 * np.pi : n_theta * 1j]
    X = np.sin(phi) * np.cos(theta)
    Y = np.sin(phi) * np.sin(theta)
    Z = np.cos(phi)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color=[0.8, 0.85, 0.95], alpha=0.3, linewidth=0)


def plot_point(ax, x, *, color="C0", label=None, s=80):
    ax.scatter(x[0], x[1], x[2], color=color, s=s)
    if label is not None:
        ax.text(x[0], x[1], x[2], f"  {label}", color=color)


def plot_tangent_vector(ax, base, vec, *, color="C0", label=None, linewidth=2):
    """
    Plots a tangent vector as a straight arrow, with the arrowhead oriented
    to be 'flat' with respect to the sphere's surface at the arrow's base.
    """
    norm_vec = np.linalg.norm(vec)
    if abs(norm_vec) < 1e-9:
        return

    # Arrow shaft
    tip = base + vec
    ax.plot([base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]], color=color, linewidth=linewidth)

    # Arrowhead
    v_norm = vec / norm_vec

    # The 'up' vector is the normal to the sphere surface at the base of the arrow
    up_vec = base / np.linalg.norm(base)

    # The 'side' vector is tangent to the sphere and perpendicular to the arrow direction
    side_vec = np.cross(v_norm, up_vec)

    # Arrowhead geometry
    head_length_ratio = 0.2
    head_width_ratio = 0.1
    head_length = norm_vec * head_length_ratio
    head_width = norm_vec * head_width_ratio

    wing_base = tip - head_length * v_norm
    wing1_tip = wing_base + head_width * side_vec
    wing2_tip = wing_base - head_width * side_vec

    # Draw arrowhead wings
    ax.plot([tip[0], wing1_tip[0]], [tip[1], wing1_tip[1]], [tip[2], wing1_tip[2]], color=color, linewidth=linewidth)
    ax.plot([tip[0], wing2_tip[0]], [tip[1], wing2_tip[1]], [tip[2], wing2_tip[2]], color=color, linewidth=linewidth)

    if label is not None:
        ax.text(tip[0], tip[1], tip[2], f"  {label}", color=color)


def plot_geodesic(ax, manifold, a, b, *, color="k", linewidth=2, n=100):
    """
    Plots a geodesic curve between two points a and b on the manifold.
    """
    gc = np.zeros((n, 3))
    ab_tangent = manifold.log(a, b)
    for i, t in enumerate(np.linspace(0, 1, n)):
        gc[i, :] = manifold.exp(a, t * ab_tangent)

    ax.plot(gc[:, 0], gc[:, 1], gc[:, 2], color=color, linewidth=linewidth)


# ---------- manifold math + calls to plotting helpers ----------


def parallel_transport_Sd(manifold, x, y, v, *, tol=1e-10):
    """Implements parallel transport for Sd.

    Equation:
        Gamma_{x->y}(v) = v - ((Log_x(y)^T v) / d(x, y)^2) (Log_x(y) + Log_y(x)),
        with d(x, y) = arccos(x^T y)
    """
    dist_xy = manifold.dist(x, y)
    if dist_xy < tol:
        return v

    log_xy = manifold.log(x, y)
    log_yx = manifold.log(y, x)

    inner_product = manifold.inner_product(x, log_xy, v)

    transported_v = v - (inner_product / (dist_xy**2)) * (log_xy + log_yx)

    return transported_v


def main():
    # S² as unit sphere in R³
    manifold = Sphere(3)

    # Base point and tangent vectors
    x = manifold.random_point()
    u = manifold.random_tangent_vector(x)  # direction for exp/log
    w = manifold.random_tangent_vector(x)  # direction to transport

    # Move along geodesic from x in direction u
    y = manifold.exp(x, u)

    # Log map at x of y
    v = manifold.log(x, y)

    # Parallel transport w from T_x S² to T_y S²
    w_y = parallel_transport_Sd(manifold, x, y, w)

    # Sanity checks
    print("‖v‖_x:", manifold.norm(x, v))
    print("dist(x, y):", manifold.dist(x, y))
    print("‖v‖_x - dist(x, y):", manifold.norm(x, v) - manifold.dist(x, y))
    print("⟨x, u⟩:", manifold.inner_product(x, x, u))
    print("⟨y, w_y⟩:", manifold.inner_product(y, y, w_y))

    # Plotting
    fig, ax = setup_axes_3d()

    plot_sphere(ax)

    plot_point(ax, x, color="C3", label="x")
    plot_point(ax, y, color="C2", label="y = Exp_x(t*u)")

    plot_geodesic(ax, manifold, x, y, color="k")

    plot_tangent_vector(ax, x, u, color="C3", label="u")
    plot_tangent_vector(ax, x, w, color="C4", label="w")
    plot_tangent_vector(ax, y, w_y, color="C2", label="transported w")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
