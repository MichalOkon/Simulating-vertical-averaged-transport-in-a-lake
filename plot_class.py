import numpy as np
import matplotlib.pyplot as plt


def plot_velocity_vector_field(n_points=10):
    x, y = np.meshgrid(np.linspace(-1, 1, n_points), np.linspace(-1, 1, n_points))
    velocity_x = y * (1 - x ** 2) / (15 + 5 * x)
    velocity_y = x * (1 - y ** 2) / (15 + 5 * x)
    plt.title("Velocity")
    plt.quiver(x, y, velocity_x, velocity_y)
    plt.show()


def plot_dispersion_vector(n_points=10):
    x, y = np.meshgrid(np.linspace(-1, 1, n_points), np.linspace(-1, 1, n_points))
    dispersion_x = 1 + np.cos(np.pi * x)
    dispersion_y = 1 + np.cos(np.pi * y)
    plt.title("Dispersion vectors")
    plt.quiver(x, y, dispersion_x, dispersion_y, color="blue")
    plt.show()


def plot_dispersion(n_points=1000, axis="x"):
    x, y = np.meshgrid(np.linspace(-1, 1, n_points), np.linspace(-1, 1, n_points))
    if axis == "x":
        dispersion_x = 1 + np.cos(np.pi * x)
        plt.imshow(dispersion_x, extent=[-1, 1, -1, 1], origin='lower')
        plt.title("$D_x$")
    elif axis == "y":
        dispersion_y = 1 + np.cos(np.pi * y)
        plt.imshow(dispersion_y, extent=[-1, 1, -1, 1], origin='lower')
        plt.title("$D_y$")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    plot_velocity_vector_field()
    plot_dispersion_vector()
    plot_dispersion(axis="x")
    plot_dispersion(axis="y")
