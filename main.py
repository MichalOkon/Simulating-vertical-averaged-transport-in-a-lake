import convergence_class as conv
import sim_class as sim

#sim = sim.Sim(10, 1e-4, 0.5, 0.5, 1e2, "Euler")
#xy = sim.simulate()
convergence_analysis = conv.Convergence(n_particles=5000, dt=0.001, x0=0.5, y0=0.5, t_end=5,
                                        scheme="Euler")
error, dt = convergence_analysis.estimate_convergence(10, 0.001)
