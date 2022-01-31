import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import sim_class as sim


def plot_velocity_vector_field(n_points=10):
    plt.clf()
    plt.figure()
    x, y = np.meshgrid(np.linspace(-1, 1, n_points), np.linspace(-1, 1, n_points))
    velocity_x = - y * (1 - x ** 2) / (15 + 5 * x)
    velocity_y = x * (1 - y ** 2) / (15 + 5 * x)
    plt.title("Velocity")
    plt.quiver(x, y, velocity_x, velocity_y)
    plt.show()


def plot_dispersion_vector(n_points=10):
    plt.clf()
    plt.figure()
    x, y = np.meshgrid(np.linspace(-1, 1, n_points), np.linspace(-1, 1, n_points))
    dispersion_x = 1 + np.cos(np.pi * x)
    dispersion_y = 1 + np.cos(np.pi * y)
    plt.title("Dispersion vectors")
    plt.quiver(x, y, dispersion_x, dispersion_y, color="blue")
    plt.show()


def plot_dispersion(n_points=1000, axis="x"):
    plt.clf()
    plt.figure()
    x, y = np.meshgrid(np.linspace(-1, 1, n_points),
                       np.linspace(-1, 1, n_points))
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


def create_animation():
    plt.clf()
    n_points = 100
    dt = 1e-4
    record_count = 10
    col = [(np.linspace(0, 255, n_points)[i], 0, 0) for i in range(n_points)]

    def init():
        scatterplot.set_offsets([] * n_points)
        scatterplot.set_color(col)
        return [scatterplot]

    def update(i, scatterplot, positions):
        plt.title(f"t = {round(i * dt * record_count, 4)}")
        plt.plot([positions[i][0][0], positions[i + 1][0][0]],
                 [positions[i][0][1], positions[i + 1][0][1]],
                 color="blue")
        scatterplot.set_offsets(positions[i + 1])
        return [scatterplot]

    simulation = sim.Sim(n_particles=n_points, dt=dt, x0=0.5, y0=0.5, t_end=1e+1, scheme="Milstein")
    positions, _ = simulation.simulate(record_count=record_count)
    print(positions[0])
    fig = plt.figure()
    plt.ion()
    scatterplot = plt.scatter([], [])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    print(len(positions))
    movement_animation = anim.FuncAnimation(
        fig, update, init_func=init, fargs=(scatterplot, positions),
        interval=10, frames=len(positions) - 1,
        blit=False, repeat=True)
    plt.show()
    return movement_animation


def create_density_graph(n_particles, t_end, xy_vector):
    plt.clf()
    fig = plt.figure()
    print(xy_vector)
    h = plt.hist2d(xy_vector[0:int(xy_vector.shape[0] / 2)].astype('float64'),
                   xy_vector[int(xy_vector.shape[0] / 2):].astype('float64'),
                   weights=2 * np.ones(int(xy_vector.shape[0] / 2)) / xy_vector.shape[0], bins=20,
                   cmap=plt.cm.BuGn_r)
    plt.xlim([-1., 1.])
    plt.ylim([-1., 1.])
    fig.colorbar(h[3])
    plt.title(
        f"Density of the particles after {t_end} second with {n_particles} particles")
    plt.show()
    plt.pause(5000)


def create_3d_density_graph(n_particles, t_end, xy_vector):
    plt.clf()
    plt.ion()
    # code adapted from
    # https://stackoverflow.com/questions/8437788/how-to-correctly-generate-a-3d-histogram-using-numpy-or-matplotlib-built-in-func
    xAmplitudes = xy_vector[0:int(xy_vector.shape[0] / 2)]
    yAmplitudes = xy_vector[int(xy_vector.shape[0] / 2):]

    x = np.array(xAmplitudes)  # turn x,y data into numpy arrays
    y = np.array(yAmplitudes)

    fig = plt.figure()  # create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')

    hist, xedges, yedges = np.histogram2d(x, y, bins=(15, 15),
                                          weights=2 * np.ones(
                                              int(xy_vector.shape[0] / 2)) /
                                                  xy_vector.shape[0])
    xpos, ypos = np.meshgrid(xedges[:-1] + xedges[1:],
                             yedges[:-1] + yedges[1:])

    xpos = xpos.flatten() / 2.
    ypos = ypos.flatten() / 2.
    zpos = np.zeros_like(xpos)

    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    dz = hist.flatten()

    cmap = cm.get_cmap('jet')
    max_height = np.max(dz)  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    plt.title(
        f"Density of the particles after {t_end} second with {n_particles} particles")
    plt.show()
    plt.pause(5000)


def plot_error(errors, timesteps, scheme, t_end, error_type):
    plt.figure()
    plt.plot(timesteps, errors)
    plt.title(f"{error_type} Error versus Timestep for {scheme} scheme")
    plt.xlabel(f"Timestep Delta t")
    plt.ylabel(f"Absolute error at T = {t_end}")
    plt.show()


if __name__ == "__main__":
    # plot_velocity_vector_field()
    # plot_dispersion_vector()
    # plot_dispersion(axis="x")
    # plot_dispersion(axis="y")

    n_points = 10000
    dt = 1e-4
    t_end = 1
    simulation = sim.Sim(n_particles=n_points, dt=dt, x0=0.5, y0=0.5,
                         t_end=t_end, scheme="Milstein")
    _, xy_vector = simulation.simulate(record_count=0)
    # anim = create_animation()
    # create_density_graph(n_points, t_end, xy_vector)
    create_3d_density_graph(n_points, t_end, xy_vector)
