import numpy as np
import plot_class


class Sim:
    def __init__(self, n_particles, dt, x0, y0, t_end, scheme):
        # Initialize coefficients
        self.n_particles = n_particles
        self.dt = dt
        self.t_end = t_end
        # Create a vector to hold coordinates of all particles, first n elements are x coordinates, the rest are y
        # coordinates
        self.xy_0 = np.zeros(2 * self.n_particles, dtype=np.longdouble)
        self.xy_0[0:self.n_particles] = x0
        self.xy_0[self.n_particles:2 * self.n_particles] = y0
        print("xy_0 vector is:", self.xy_0)
        self.scheme = scheme

    @staticmethod
    def dispersion(coordinate_vector):
        # Calculate Dx and Dy dispersion coefficients for coordinates in the coordinate (xy) vector
        return 1 + np.cos(np.pi * coordinate_vector)

    def depth(self, coordinate_vector):
        # Calculate depth for coordinates in the coordinate vector. Ofc there is one depth for one pair of (x, y)
        depth = np.zeros(2 * self.n_particles, dtype=np.longdouble)
        depth[0:self.n_particles] = 15 + 5 * coordinate_vector[0:self.n_particles]
        depth[self.n_particles:2 * self.n_particles] = np.copy(depth[0:self.n_particles])
        return depth

    def velocity(self, coordinate_vector):
        # Calculate velocity vector with first n elements being u and rest being v
        x = coordinate_vector[0:self.n_particles]
        y = coordinate_vector[self.n_particles:2 * self.n_particles]
        velocity = np.zeros(2 * self.n_particles, dtype=np.longdouble)
        velocity[0:self.n_particles] = - y * (1 - x ** 2)
        velocity[self.n_particles:2 * self.n_particles] = x * (1 - y ** 2)
        return velocity / self.depth(coordinate_vector)

    def g_function(self, coordinate_vector):
        # Calculate g function value
        return np.sqrt(2 * self.dispersion(coordinate_vector))

    def g_function_derivative(self, coordinate_vector):
        # Calculate derivative of g function
        return self.dispersion_derivative(coordinate_vector) / self.g_function(coordinate_vector)

    @staticmethod
    def dispersion_derivative(coordinate_vector):
        # Calculate derivative of dispersion coefficients
        return - np.pi * np.sin(np.pi * coordinate_vector)

    def depth_derivative(self):
        # Calculate derivative of depth
        depth_derivative = np.zeros(2 * self.n_particles, dtype=np.longdouble)
        depth_derivative[0:self.n_particles] = 5.0
        return depth_derivative

    def hd_derivative(self, coordinate_vector):
        # Calculate derivative of depth*dispersion with product rule
        return self.depth(coordinate_vector) * self.dispersion_derivative(coordinate_vector) + \
               self.dispersion(coordinate_vector) * self.depth_derivative()

    def simulate(self):
        xy_vector = np.copy(self.xy_0)
        t = 0
        # This is a vector that holds random variables for all particles in both directions at t
        w_old = np.random.normal(loc=0.0, scale=1.0, size=2 * self.n_particles)
        print("w_old is", w_old)
        n = 0
        while t < self.t_end:
            # This is a vector that holds random variables for all particles in both directions at t+1
            w_new = np.random.normal(loc=0.0, scale=1.0, size=2 * self.n_particles)
            # print("w_new is", w_new)
            dw = (w_new - w_old) * self.dt
            if self.scheme == "Euler":
                dxy = (self.velocity(xy_vector) + self.hd_derivative(xy_vector) / self.depth(xy_vector)) * self.dt \
                      + self.g_function(xy_vector) * dw
            elif self.scheme == "Milstein":
                dxy = (self.velocity(xy_vector) + self.hd_derivative(xy_vector) / self.depth(xy_vector)) * self.dt \
                      + self.g_function(xy_vector) * dw + 0.5 * self.g_function(xy_vector) * \
                      self.g_function_derivative(xy_vector) * (dw ** 2 - self.dt)
            else:
                raise Warning("The scheme should be Euler or Milstein")
            xy_vector += dxy
            if n % 100 == 0:
                plot_class.plot_particle_movement(xy_vector)
            # print("xy_vector is:", xy_vector)
            w_old = w_new
            t += self.dt
            n += 1
        return xy_vector
