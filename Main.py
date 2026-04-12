from Simulation import Simulation

sim = Simulation(n_cells=100, length=1.0, dt=0.005, n_steps=3000)
sim.run()
sim.plot_phase_space("Final")
sim.plot_results()
