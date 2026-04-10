from Grid import Grid
from Species import Species
import numpy as np
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, n_cells, length, dt, n_steps):
        self.grid = Grid(n_cells, length)
        self.dt = dt
        self.n_steps = n_steps
        self.species = None

    def run(self):
        self.species_e = Species('-e', 0.01, -1/100000, 100000) #electron example
        self.species_I = Species('I', 1, 1/100000, 100000) #ion example
        self.species_e.initialize_harris_particles(self.grid.nx * self.grid.dx, 1, 0.1, 0.2, self.grid)
        self.species_I.initialize_harris_particles(self.grid.nx * self.grid.dx, 1, 0.1, 0.2, self.grid)
        self.species_e.deposit_sources(self.grid)
        self.species_I.deposit_sources(self.grid)

        for step in range(self.n_steps):
            # Deposit charge and current from all species onto the grid
            self.species_e.deposite_charge_currents(self.grid)
            self.species_I.deposite_charge_currents(self.grid)

            # Solve Maxwell's equations to update fields
            self.grid.solve_fields(self.dt)

            # Push particles of each species using the updated fields
            self.species_e.push(self.grid, self.dt)
            self.species_I.push(self.grid, self.dt)

    def plot_results(self):
        '''Plots a Phase space plot of the particles position/velocity'''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Plot 1: Position (x) vs Velocity (vx) ---
        # Shows the thermal spread and any electrostatic acceleration
        ax1.scatter(self.species_e.x, self.species_e.vx, s=0.5, alpha=0.2, color='blue', label='Electrons')
        ax1.scatter(self.species_I.x, self.species_I.vx, s=0.5, alpha=0.2, color='red', label='Ions')
        
        ax1.set_title("Phase Space: x vs $v_x$ (Longitudinal)")
        ax1.set_xlabel("Position (x)")
        ax1.set_ylabel("Velocity ($v_x$)")
        ax1.set_xlim(self.grid.x_min, self.grid.x_max)
        ax1.set_ylim(-0.3, 0.3) # Adjust based on your v_th
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right')

        # --- Plot 2: Position (x) vs Velocity (vz) ---
        # This is the "Current" plot. You should see the drift offset here!
        ax2.scatter(self.species_e.x, self.species_e.vz, s=0.5, alpha=0.2, color='blue')
        ax2.scatter(self.species_I.x, self.species_I.vz, s=0.5, alpha=0.2, color='red')
        
        ax2.set_title("Phase Space: x vs $v_z$ (Out-of-Plane Drift)")
        ax2.set_xlabel("Position (x)")
        ax2.set_ylabel("Velocity ($v_z$)")
        ax2.set_xlim(self.grid.x_min, self.grid.x_max)
        ax2.set_ylim(-1.0, 1.0) # Set wide enough to see u_drift
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig("phase_space_plots.png", dpi=300)
        plt.show()

