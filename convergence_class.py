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

    @staticmethod
    def calculate_strong_error(xy, xy_accurate):
        strong_error = np.linalg.norm(xy - xy_accurate, axis=0).mean()
        print(strong_error)
        return strong_error

    @staticmethod
    def calculate_weak_error(xy, xy_accurate):
        weak_error = np.linalg.norm((xy - xy_accurate).mean(axis=1))
        print(weak_error)
        return weak_error

    def estimate_convergence(self, n_samples, dt_ratio):
        print(f"Running convergence simulation 1")
        sim_accurate = sim.Sim(self.n_particles, self.dt * dt_ratio, self.x0, self.y0, self.t_end,
                               self.scheme)
        _, accurate_last_positon = sim_accurate.simulate()
        xy_accurate = np.array([accurate_last_positon [:self.n_particles], accurate_last_positon [self.n_particles:]])
        dt_log = np.zeros(n_samples)
        weak_error_log = np.zeros(n_samples)
        strong_error_log = np.zeros(n_samples)
        for i in range(0, n_samples):
            print(f"Running convergence simulation {i + 2}")
            dt = self.dt * (i + 1) / n_samples
            dummy_sim = sim.Sim(self.n_particles, dt, self.x0, self.y0, self.t_end, self.scheme)
            position_data, last_position = dummy_sim.simulate()
            xy_positions = np.array([last_position[:self.n_particles], last_position[self.n_particles:]])
           # xy_positions = [[last_position[i], last_position[int(self.n_particles / 2) + i]] for i in
            #                 range(int(self.n_particles / 2))]
            print(xy_positions)
            print(xy_accurate)
            strong_error_log[i] = self.calculate_strong_error(xy_positions,xy_accurate )
            weak_error_log[i] = self.calculate_weak_error(xy_positions, xy_accurate)
            dt_log[i] = dt

        plot_error(weak_error_log, dt_log, self.scheme, self.t_end, self.n_particles, "Weak")
        plot_error(strong_error_log, dt_log, self.scheme, self.t_end, self.n_particles, "Strong")

        # print(weak_error)
        # print(strong_error)
        return strong_error_log, weak_error_log, dt_log
