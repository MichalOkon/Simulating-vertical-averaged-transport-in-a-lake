import convergence_class as conv
import sim_class as sim
import numpy as np

# sim = sim.Sim(10, 1e-4, 0.5, 0.5, 1e2, "Euler")
# xy = sim.simulate()
convergence_analysis = conv.Convergence(n_particles=10000, dt=1, x0=0.5, y0=0.5, t_end=1,
                                        scheme="Euler")
strong_error, weak_error, dt = convergence_analysis.estimate_convergence(200, 0.001)
