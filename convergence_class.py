import numpy as np
import sim_class as sim


class Convergence:
    def __init__(self, n_particles, dt, x0, y0, t_end, scheme):
        self.n_particles = n_particles
        self.dt = dt
        self.t_end = t_end
        self.x0 = x0
        self.y0 = y0
        self.scheme = scheme

    def weak_convergence(self, n_samples, dt_ratio):
        sim_accurate = sim.Sim(self.n_particles, self.dt*dt_ratio, self.x0, self.y0, self.t_end, self.scheme)
        xy_accurate = sim_accurate.simulate()
        xy_log = np.zeros((n_samples, 2 * self.n_particles))
        dt_log = np.zeros(n_samples)
        for i in range(0, n_samples):
            dt = self.dt * (i+1) / n_samples
            dummy_sim = sim.Sim(self.n_particles, dt, self.x0, self.y0, self.t_end, self.scheme)
            xy = dummy_sim.simulate()
            xy_log[i] = xy
            dt_log[i] = dt
        weak_error = np.mean(xy_log, axis=1) - np.mean(xy_accurate)
        return weak_error, dt_log


