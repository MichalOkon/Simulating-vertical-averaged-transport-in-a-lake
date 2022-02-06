import numpy as np

import sim_class as sim
from plot_class import plot_error


class Convergence:
    def __init__(self, n_particles, dt, x0, y0, t_end, scheme, domain_behavior="clip"):
        self.n_particles = n_particles
        self.dt = dt
        self.t_end = t_end
        self.x0 = x0
        self.y0 = y0
        self.scheme = scheme
        self.domain_behavior = domain_behavior

    @staticmethod
    def calculate_strong_error(xy, xy_accurate, excluded_particles):
        strong_error = np.linalg.norm(
            np.delete(xy, excluded_particles, 1) - np.delete(xy_accurate, excluded_particles, 1), axis=0).mean()
        print(strong_error)
        return strong_error

    @staticmethod
    def calculate_weak_error(xy, xy_accurate, excluded_particles):
        weak_error = np.linalg.norm(
            (np.delete(xy, excluded_particles, 1) - np.delete(xy_accurate, excluded_particles, 1)).mean(axis=1))
        print(weak_error)
        return weak_error

    def estimate_convergence(self, n_samples, dt_ratio):
        print(f"Running convergence simulation 1")
        sim_conv = sim.Sim(self.n_particles, self.dt, self.x0, self.y0, self.t_end,
                               self.scheme, domain_behavior=self.domain_behavior)
        _, accurate_last_positon, _ = sim_conv.simulate()
        xy_accurate = np.array([accurate_last_positon[:self.n_particles], accurate_last_positon[self.n_particles:]])
        dt_log = np.zeros(n_samples)
        weak_error_log = np.zeros(n_samples)
        strong_error_log = np.zeros(n_samples)
        for i in range(0, n_samples):
            print(f"Running convergence simulation {i + 2}")
            sim_conv.set_step_size((i+1)*dt_ratio)
            position_data, last_position, excluded_particles = sim_conv.simulate()
            xy_positions = np.array([last_position[:self.n_particles], last_position[self.n_particles:]])
            # xy_positions = [[last_position[i], last_position[int(self.n_particles / 2) + i]] for i in
            #                 range(int(self.n_particles / 2))]
           # print(xy_positions)
            #print(f"Number of particles out: {len(excluded_particles)}")
            strong_error_log[i] = self.calculate_strong_error(xy_positions, xy_accurate, excluded_particles)
            weak_error_log[i] = self.calculate_weak_error(xy_positions, xy_accurate, excluded_particles)
            dt_log[i] = self.dt * sim_conv.step_size

        plot_error(weak_error_log[:-1], dt_log[:-1], self.scheme, self.t_end, self.n_particles, "Weak")
        plot_error(strong_error_log[:-1], dt_log[:-1], self.scheme, self.t_end, self.n_particles, "Strong")

        # print(weak_error)
        # print(strong_error)

        return strong_error_log, weak_error_log, dt_log
