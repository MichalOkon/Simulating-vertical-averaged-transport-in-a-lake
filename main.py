import sim_class as sim
import convergence_class as conv

sim = sim.Sim(2, 1e-32, 0.5, 0.5, 1e2, "Euler")
xy = sim.simulate()
# convergence_analysis = conv.Convergence(10, 1e-8, 0.5, 0.5, 1e2, "Euler")
# error, dt = convergence_analysis.weak_convergence(10, 1)
