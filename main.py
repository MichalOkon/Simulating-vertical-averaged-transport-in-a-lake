import convergence_class as conv
import sim_class as sim

# sim = sim.Sim(10, 1e-4, 0.5, 0.5, 1e2, "Euler")
# xy = sim.simulate()
convergence_analysis = conv.Convergence(n_particles=10000, dt=1, x0=0.5, y0=0.5, t_end=1,
                                        scheme="Euler", domain_behavior="clip")
strong_error, weak_error, dt = convergence_analysis.estimate_convergence(200, 0.001)

#sim = sim.Sim(n_particles=10000, dt=0.005, x0=0.5, y0=0.5, t_end=1, scheme="Euler", domain_behavior="catch")
#_, _, excluded_particles = sim.simulate()
#print(len(excluded_particles)/10000)