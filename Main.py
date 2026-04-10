from Simulation import Simulation

sim = Simulation(n_cells=100, length=1.0, dt=0.001, n_steps=100)
sim.run()
sim.plot_results()