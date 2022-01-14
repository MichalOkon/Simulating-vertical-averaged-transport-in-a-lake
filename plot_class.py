import numpy as np
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import sim_class as sim


def plot_velocity_vector_field(n_points=10):
    plt.figure()
    x, y = np.meshgrid(np.linspace(-1, 1, n_points), np.linspace(-1, 1, n_points))
    velocity_x = y * (1 - x ** 2) / (15 + 5 * x)
    velocity_y = x * (1 - y ** 2) / (15 + 5 * x)
    plt.title("Velocity")
    plt.quiver(x, y, velocity_x, velocity_y)
    plt.draw()


def plot_dispersion_vector(n_points=10):
    plt.figure()
    x, y = np.meshgrid(np.linspace(-1, 1, n_points), np.linspace(-1, 1, n_points))
    dispersion_x = 1 + np.cos(np.pi * x)
    dispersion_y = 1 + np.cos(np.pi * y)
    plt.title("Dispersion vectors")
    plt.quiver(x, y, dispersion_x, dispersion_y, color="blue")
    plt.draw()


def plot_dispersion(n_points=1000, axis="x"):
    plt.figure()
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
    plt.draw()


def create_graph():
    n_points = 10
    col = [(np.arange(0, 255, n_points)[i], 0, 0) for i in range(n_points)]

    def init():
        scatterplot.set_offsets([] * n_points)
        scatterplot.set_color(col)
        return [scatterplot]

    def update(i, scatterplot, positions):
        scatterplot.set_offsets(positions[i])

        return [scatterplot]

    simulation = sim.Sim(n_particles=n_points, dt=1e-4, x0=0.1, y0=0, t_end=1e+1, scheme="Euler")
    positions = simulation.simulate(record_count=10)

    fig = plt.figure()
    plt.ion()
    scatterplot = plt.scatter([], [])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    print(len(positions))
    movement_animation = anim.FuncAnimation(
        fig, update, init_func=init, fargs=(scatterplot, positions), interval=1, frames=len(positions),
        blit=True, repeat=True)
    plt.show()
    return movement_animation


if __name__ == "__main__":
    plt.ion()
    plot_velocity_vector_field()
    plot_dispersion_vector()
    plot_dispersion(axis="x")
    plot_dispersion(axis="y")
    anim = create_graph()
