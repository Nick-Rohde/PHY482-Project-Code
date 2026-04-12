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
        self.T = []
        self.V = []
        self.Ex_snapshots = []
        self.rho_snapshots = []
        self.shot_steps = []


    def run(self):
        shot_interval = self.n_steps // 10

        self.species_e1 = Species('-e', 1, -1, 100000) #electron example
        self.species_e2 = Species('-e', 1, -1, 100000) #electron example
        #self.species_e2 = Species('I', 1, 1, 100000) #ion example

        #L_sheet = self.grid.nx * self.grid.dx * 0.1  # Example thickness for the current sheet

        #self.species_e1.initialize_harris_particles(L_sheet, 1, 0.1, -0.2, self.grid)
        #self.species_e2.initialize_harris_particles(L_sheet, 1, 0.1, 0.5, self.grid)
        
        self.species_e1.initialize_two_stream(0.02, 0.2, self.grid)
        self.species_e2.initialize_two_stream(0.02, -0.2, self.grid)

        self.species_e1.deposit_sources(self.grid)
        self.species_e2.deposit_sources(self.grid)
        self.grid.init_fields(B0=1.0)
        self.plot_phase_space("Initial")
        for step in range(self.n_steps):
            # Deposit charge and current from all species onto the grid
            self.grid.rho.fill(0)
            self.grid.Jy.fill(0)
            self.grid.Jx.fill(0)
            self.grid.Jz.fill(0)

            self.species_e1.deposite_charge_currents(self.grid)
            self.species_e2.deposite_charge_currents(self.grid)

            # Solve Maxwell's equations to update fields
            self.grid.solve_fields(self.dt)

            # Push particles of each species using the updated fields
            self.species_e1.push(self.grid, self.dt)
            self.species_e2.push(self.grid, self.dt)

            #calculate KE and PE for plots!
            #mag_energy = np.sum(self.grid.Bx**2 + self.grid.By**2 + self.grid.Bz**2) * self.grid.dx
            #self.V.append(mag_energy)

            elec_energy = np.sum(self.grid.Ex**2) * self.grid.dx
            self.V.append(elec_energy)

            # Kinetic Energy: 0.5 * m * v^2
            ke_e = 0.5 * self.species_e1.mass * np.sum(self.species_e1.vx**2 + self.species_e1.vy**2 + self.species_e1.vz**2)
            ke_i = 0.5 * self.species_e2.mass * np.sum(self.species_e2.vx**2 + self.species_e2.vy**2 + self.species_e2.vz**2)
            self.T.append(ke_e + ke_i)

            if step % shot_interval == 0 or step == self.n_steps - 1:
                self.Ex_snapshots.append(self.grid.Ex.copy())
                self.rho_snapshots.append(self.grid.rho.copy())
                self.shot_steps.append(step)
                self.plot_phase_space(f"Step {step}")

            

    def plot_phase_space(self, titlestring):
        '''Plots a Phase space plot of the particles position/velocity'''
        '''
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
        # Shows the thermal spread and any electrostatic acceleration
        ax1.scatter(self.species_e1.x, self.species_e1.vx, s=0.5, alpha=0.2, color='blue', label='Electrons 1')
        ax1.scatter(self.species_e2.x, self.species_e2.vx, s=0.5, alpha=0.2, color='red', label='Electrons 2')
        
        ax1.set_title("Phase Space: x vs $v_x$ (Longitudinal) at " + titlestring)
        ax1.set_xlabel("Position (x)")
        ax1.set_ylabel("Velocity ($v_x$)")
        ax1.set_xlim(self.grid.x_min, self.grid.x_max)
        ax1.set_ylim(-0.3, 0.3) # Adjust based on your v_th
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right')

        # Shows the out-of-plane drift and any acceleration from reconnection electric fields
        ax2.scatter(self.species_e1.x, self.species_e1.vz, s=0.5, alpha=0.2, color='blue')
        ax2.scatter(self.species_e2.x, self.species_e2.vz, s=0.5, alpha=0.2, color='red')
        
        ax2.set_title("Phase Space: x vs $v_z$ (Out-of-Plane Drift) at " + titlestring)
        ax2.set_xlabel("Position (x)")
        ax2.set_ylabel("Velocity ($v_z$)")
        ax2.set_xlim(self.grid.x_min, self.grid.x_max)
        ax2.set_ylim(-1.0, 1.0) # Set wide enough to see u_drift
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig("Plots/phase_space_plots_" + titlestring + ".pdf", dpi=300)
        plt.show()
        '''
        fig = plt.figure(figsize=(8,6))
        plt.scatter(self.species_e1.x, self.species_e1.vx, s=0.5, alpha=0.2, color='blue', label='Electrons 1')
        plt.scatter(self.species_e2.x, self.species_e2.vx, s=0.5, alpha=0.2, color='red', label='Electrons 2')
        plt.title("Phase Space: x vs $v_x$ (Longitudinal) at " + titlestring)
        plt.xlabel("Position (x)")
        plt.ylabel("Velocity ($v_x$)")
        plt.xlim(self.grid.x_min, self.grid.x_max)
        plt.ylim(-0.8, 0.8) # Adjust based on your v_th
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("Plots/phase_space_plots_" + titlestring + ".pdf", dpi=300)
        plt.close(fig)

    def plot_results(self):
        ''' plots the results determined in the main loop'''

        #energy plot
        fig = plt.figure(figsize=(8,6))
        plt.plot(self.T, label='Kinetic Energy')
        plt.plot(self.V, label='Magnetic Energy')
        plt.xlabel('Time Step')
        plt.ylabel('Energy')
        plt.title('Energy Evolution')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("Plots/energy_evolution.pdf", dpi=300)
        plt.close(fig)

        # Plot snapshots of By and Jz
        x_axis = np.linspace(self.grid.x_min, self.grid.x_max, self.grid.nx)
        for i, step in enumerate(self.shot_steps):
            fig = plt.figure(figsize=(12,5))
            plt.subplot(1,2,1)
            plt.plot(x_axis, self.Ex_snapshots[i], label='Ex')
            plt.title(f'Ex at Step {step}')
            plt.xlabel('x')
            plt.ylabel('Ex')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()

            plt.subplot(1,2,2)
            plt.plot(x_axis, self.rho_snapshots[i], label='Rho')
            plt.title(f'Rho at Step {step}')
            plt.xlabel('x')
            plt.ylabel('Rho')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()

            plt.tight_layout()
            plt.savefig(f'Plots/field_snapshots_step_{step}.pdf', dpi=300)
            plt.close(fig)

