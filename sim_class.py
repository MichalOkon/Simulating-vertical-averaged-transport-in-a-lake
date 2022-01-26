import numpy as np
import plot_class
import matplotlib.pyplot as plt
from numpy.random import default_rng


class Sim:
    pi = np.pi

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
        self.pi = np.pi
        self.dispersion_sin = 0
        self.dispersion_cos = 0

    def dispersion(self, coordinate_vector):
        # Calculate Dx and Dy dispersion coefficients for coordinates in the coordinate (xy) vector
        return 1 + self.dispersion_cos

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
        return self.dispersion_derivative() / self.g_function(coordinate_vector)

    def dispersion_derivative(self):
        # Calculate derivative of dispersion coefficients
        return - self.pi * self.dispersion_sin

    def depth_derivative(self):
        # Calculate derivative of depth
        depth_derivative = np.zeros(2 * self.n_particles, dtype=np.longdouble)
        depth_derivative[0:self.n_particles] = 5.0
        return depth_derivative

    def hd_derivative(self, coordinate_vector):
        # Calculate derivative of depth*dispersion with product rule
        return self.depth(coordinate_vector) * self.dispersion_derivative() + \
               self.dispersion(coordinate_vector) * self.depth_derivative()

    def calculate_cos_sin(self, xy_vector):
        inner = np.pi * xy_vector
        self.dispersion_sin = np.sin(inner)
        self.dispersion_cos = np.cos(inner)

    def simulate(self, record_count=1):
        xy_vector = np.copy(self.xy_0)
        x_coords = xy_vector[0:int(xy_vector.shape[0] / 2)]
        y_coords = xy_vector[int(xy_vector.shape[0] / 2):]
        t = 0
        # This is a vector that holds random variables for all particles in both directions at t
        w_old = 0
        print("w_old is", w_old)
        n = 0
        position_data = [[[x_coords[i], y_coords[i]] for i in range(x_coords.shape[0])]]

        rng = default_rng()
        while t < self.t_end:
            self.calculate_cos_sin(xy_vector)

            dw = np.sqrt(self.dt) * rng.standard_normal(2 * self.n_particles)
            if self.scheme == "Euler":
                dxy = (self.velocity(xy_vector) + (self.hd_derivative(xy_vector) / self.depth(xy_vector))) * self.dt \
                      + self.g_function(xy_vector) * dw
            elif self.scheme == "Milstein":
                dxy = (self.velocity(xy_vector) + self.hd_derivative(xy_vector) / self.depth(xy_vector)) * self.dt \
                      + self.g_function(xy_vector) * dw + 0.5 * self.g_function(xy_vector) * \
                      self.g_function_derivative(xy_vector) * (dw ** 2 - self.dt)
            else:
                raise Warning("The scheme should be Euler or Milstein")
            xy_vector += dxy
            if record_count != 0 and n % record_count == 0:
                position_data.append([[x_coords[i], y_coords[i]] for i in range(x_coords.shape[0])])
            # print("xy_vector is:", xy_vector)
            # w_old = w_new
            t += self.dt
            n += 1
        return position_data, xy_vector
