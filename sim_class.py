import numpy as np
from numpy.random import default_rng


class Sim:
    pi = np.pi

    def __init__(self, n_particles, dt, x0, y0, t_end, scheme, domain_behavior="clip", step_size=1):
        # Initialize coefficients
        self.n_particles = n_particles
        self.dt = dt
        self.t_end = t_end
        self.n_intervals = int(t_end // dt)
        self.step_size = step_size
        # Create a vector to hold coordinates of all particles, first n elements are x coordinates, the rest are y
        # coordinates
        self.xy_0 = np.zeros(2 * self.n_particles, dtype=np.longdouble)
        self.xy_0[0:self.n_particles] = x0
        self.xy_0[self.n_particles:2 * self.n_particles] = y0
        # print("xy_0 vector is:", self.xy_0)
        self.scheme = scheme
        self.pi = np.pi
        self.dispersion_sin = 0
        self.dispersion_cos = 0

        self.xy_vector = np.copy(self.xy_0)
        self.x_coords = self.xy_vector[0:int(self.xy_vector.shape[0] / 2)]
        self.y_coords = self.xy_vector[int(self.xy_vector.shape[0] / 2):]
        self.excluded_particles = []

        # Either catch or clip or ignore
        self.domain_behavior = domain_behavior

        self.wiener_process = default_rng(1234).normal(0, np.sqrt(self.dt),
                                                      (self.n_intervals, 2 * self.n_particles))
        print(self.wiener_process)

    def initialize_simulation(self):
        # Creating the vectors holding the current states of the simulation
        self.xy_vector = np.copy(self.xy_0)
        self.x_coords = self.xy_vector[0:int(self.xy_vector.shape[0] / 2)]
        self.y_coords = self.xy_vector[int(self.xy_vector.shape[0] / 2):]
        self.excluded_particles = []

    def set_step_size(self, step_size):
        self.step_size = step_size

    def get_dw(self, step):
        if step * self.step_size > self.n_intervals:
            return self.wiener_process[self.step_size * (step-1):].sum(axis=0)
        if step == 0:
            return self.wiener_process[0]
        return self.wiener_process[self.step_size * (step-1):step * self.step_size].sum(axis=0)

    def dispersion(self):
        # Calculate Dx and Dy dispersion coefficients for coordinates in the coordinate (xy) vector
        return 1 + self.dispersion_cos

    def depth(self):
        # Calculate depth for coordinates in the coordinate vector. Ofc there is one depth for one pair of (x, y)
        depth = np.zeros(2 * self.n_particles, dtype=np.longdouble)
        depth[0:self.n_particles] = 15 + 5 * self.xy_vector[0:self.n_particles]
        depth[self.n_particles:2 * self.n_particles] = np.copy(depth[0:self.n_particles])
        return depth

    def velocity(self):
        # Calculate velocity vector with first n elements being u and rest being v
        x = self.xy_vector[0:self.n_particles]
        y = self.xy_vector[self.n_particles:2 * self.n_particles]
        velocity = np.zeros(2 * self.n_particles, dtype=np.longdouble)
        velocity[0:self.n_particles] = - y * (1 - x ** 2)
        velocity[self.n_particles:2 * self.n_particles] = x * (1 - y ** 2)
        return velocity / self.depth()

    def g_function(self):
        # Calculate g function value
        return np.sqrt(2 * self.dispersion())

    def g_function_derivative(self):
        # Calculate derivative of g function
        return self.dispersion_derivative() / self.g_function()

    def dispersion_derivative(self):
        # Calculate derivative of dispersion coefficients
        return - self.pi * self.dispersion_sin

    def depth_derivative(self):
        # Calculate derivative of depth
        depth_derivative = np.zeros(2 * self.n_particles, dtype=np.longdouble)
        depth_derivative[0:self.n_particles] = 5.0
        return depth_derivative

    def hd_derivative(self):
        # Calculate derivative of depth*dispersion with product rule
        return self.depth() * self.dispersion_derivative() + \
               self.dispersion() * self.depth_derivative()

    def calculate_cos_sin(self):
        inner = np.pi * self.xy_vector
        self.dispersion_sin = np.sin(inner)
        self.dispersion_cos = np.cos(inner)

    def catch_escaping_particles(self, catch=True):
        # Checks if any of the particles left the domain, sets their coordinates to (1, 1) (stable point)
        # and returns the indices of  particles that left the domain
        x_out_of_domain = np.where(np.logical_or(self.x_coords < -1, self.x_coords > 1))
        y_out_of_domain = np.where(np.logical_or(self.y_coords < -1, self.y_coords > 1))
        xy_out_of_domain = np.union1d(x_out_of_domain, y_out_of_domain)

        # print(f"Out of domain{len(xy_out_of_domain)}")
        self.excluded_particles.extend(xy_out_of_domain)
        particles_out = len(xy_out_of_domain)

        if catch:
            self.x_coords[xy_out_of_domain] = 1
            self.y_coords[xy_out_of_domain] = 1

        return particles_out

    def simulate(self, record_count=0):
        t = 0
        n = 0
        self.initialize_simulation()
        position_data = [[[self.x_coords[i], self.y_coords[i]] for i in range(self.x_coords.shape[0])]]

        number_of_iterations = self.n_intervals // self.step_size
        print(f"n intervals = {number_of_iterations}")
        time_step = self.dt * self.step_size
        while n < number_of_iterations:

            # print(time_step)
            self.calculate_cos_sin()

            dw = self.get_dw(n)
            # print(dw)
            if self.scheme == "Euler":
                dxy = (self.velocity() + (self.hd_derivative() / self.depth())) * time_step \
                      + self.g_function() * dw
            elif self.scheme == "Milstein":
                dxy = (self.velocity() + self.hd_derivative() / self.depth()) * time_step \
                      + self.g_function() * dw + 0.5 * self.dispersion_derivative() * (
                              dw ** 2 - time_step)
            else:
                raise Warning("The scheme should be Euler or Milstein")
            self.xy_vector += dxy
            if record_count != 0 and n % record_count == 0:
                position_data.append(
                    [[self.x_coords[i], self.y_coords[i]] for i in range(self.x_coords.shape[0])])


            if self.domain_behavior == "clip":
                self.xy_vector = np.clip(self.xy_vector, -1, 1)
            elif self.domain_behavior == "catch":
                self.catch_escaping_particles()

            #print(self.xy_vector)
            t += time_step
            n += 1

        print(f"excluded particles: {self.excluded_particles}")
        return position_data, self.xy_vector, self.excluded_particles
