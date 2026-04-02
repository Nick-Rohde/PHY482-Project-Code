import numpy as np

class Grid:
    def __init__(self, n_cells, length):
        self.nx = n_cells
        self.dx = length / n_cells
        
        # Field arrays (on the Yee lattice)
        self.E = np.zeros(n_cells)
        self.B = np.zeros(n_cells)
        self.rho = np.zeros(n_cells) # Charge density
        self.J = np.zeros(n_cells)   # Current density

    def solve_fields(self, dt):
        """Solve Maxwell's equations."""
        # Update E and B based on J and rho using a simple finite-difference time-domain (FDTD) method

        for i in range(1, self.nx - 1):
            # uopdate B to dt/2 using E for half step
            self.B[i] += - (self.E[i+1] - self.E[i]) * dt / (2 * self.dx)

            # update E to dt using B for full step
            self.E[i] += (self.B[i] - self.B[i-1]) * dt / self.dx - self.J[i] * dt / self.dx

            #update B from dt/2 to dt for half step part 2
            self.B[i] += - (self.E[i+1] - self.E[i]) * dt / (2 * self.dx)

    def interpolate_fields(self, x):
        """Interpolate E and B fields at particle position x."""
        cell_index = int(x / self.dx)
        h = cell_index - x / self.dx
        W_r = 1 - abs(h)
        W_l = abs(h)
        E_interp = W_l * self.E[cell_index] + W_r * self.E[cell_index + 1]
        B_interp = W_l * self.B[cell_index] + W_r * self.B[cell_index + 1]
        return E_interp, B_interp