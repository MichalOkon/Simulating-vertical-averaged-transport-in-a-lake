import numpy as np

import sim_class as sim
from plot_class import plot_error


class Convergence:
    def __init__(self, n_particles, dt, x0, y0, t_end, scheme):
        self.n_particles = n_particles
        self.dt = dt
        self.t_end = t_end
        self.x0 = x0
        self.y0 = y0
        self.scheme = scheme

    def calculate_strong_error(self, x_log, y_log, x_accurate, y_accurate):
        strong_error_x = np.mean(np.abs(x_log - x_accurate), axis=1)
        strong_error_y = np.mean(np.abs(y_log - y_accurate), axis=1)
        strong_error_xy = np.stack((strong_error_x, strong_error_y), axis=1)
        print(strong_error_xy)
        strong_error = np.linalg.norm(strong_error_xy, axis=1)
        return strong_error

    def calculate_weak_error(self, x_log, y_log, x_accurate, y_accurate):
        weak_error_x = np.mean(np.abs(x_log), axis=1) - np.mean(np.abs(x_accurate))
        weak_error_y = np.mean(np.abs(y_log), axis=1) - np.mean(np.abs(y_accurate))
        print("mean x", np.mean(np.abs(x_log), axis=1))
        print("mena x accurate", np.mean(np.abs(x_accurate)))
        weak_error_xy = np.stack((weak_error_x, weak_error_y), axis=1)
        weak_error = np.linalg.norm(weak_error_xy, axis=1)

        return weak_error

    def estimate_convergence(self, n_samples, dt_ratio):
        print(f"Running convergence simulation 1")
        sim_accurate = sim.Sim(self.n_particles, self.dt * dt_ratio, self.x0, self.y0, self.t_end,
                               self.scheme)
        xy_accurate = np.reshape(sim_accurate.simulate()[1], (1, -1))
        xy_log = np.zeros((n_samples, 2 * self.n_particles))
        dt_log = np.zeros(n_samples)
        for i in range(0, n_samples):
            print(f"Running convergence simulation {i + 1}")
            dt = self.dt * (i + 1) / n_samples
            dummy_sim = sim.Sim(self.n_particles, dt, self.x0, self.y0, self.t_end, self.scheme)
            position_data, last_position = dummy_sim.simulate()
            xy_log[i] = last_position
            dt_log[i] = dt

        # print(xy_accurate)
        # print(xy_log)
        x_accurate = xy_accurate[0, : int(xy_accurate.shape[1] / 2)]
        y_accurate = xy_accurate[0, int(xy_accurate.shape[1] / 2):]
        x_log = xy_log[:, : int(xy_log.shape[1] / 2)]
        y_log = xy_log[:, int(xy_log.shape[1] / 2):]
        print(x_accurate)
        print(x_log)
        print(f"absolute = {np.abs(x_log - x_accurate)}")
        weak_error = self.calculate_weak_error(x_log, y_log, x_accurate, y_accurate)
        strong_error = self.calculate_strong_error(x_log, y_log, x_accurate, y_accurate)

        plot_error(weak_error, dt_log, self.scheme, self.t_end, "Weak")
        plot_error(strong_error, dt_log, self.scheme, self.t_end, "Strong")

        # print(weak_error)
        # print(strong_error)
        return strong_error, dt_log
