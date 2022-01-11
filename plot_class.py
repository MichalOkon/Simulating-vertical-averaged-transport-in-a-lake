import numpy as np
import matplotlib.pyplot as plt


def plot_velocity_vector_field():
    x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    velocity_x = y * (1 - x ** 2) / (15 + 5 * x)
    velocity_y = x * (1 - y ** 2) / (15 + 5 * x)
    print(x)
    print(y)
    plt.quiver(x, y, velocity_x, velocity_y)
    plt.show()


if __name__ == "__main__":
    plot_velocity_vector_field()
