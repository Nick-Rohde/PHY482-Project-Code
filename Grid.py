import numpy as np

class Grid:
    def __init__(self, n_cells, length):
        self.nx = n_cells
        self.dx = length / n_cells
        self.x_min = 0.0
        self.x_max = length
        
        # Electric fields (on the Yee lattice)
        self.Ex = np.zeros(n_cells)
        self.Ey = np.zeros(n_cells)
        self.Ez = np.zeros(n_cells)

        # Magnetic fields (also on the Yee lattice)
        self.Bx = np.zeros(n_cells)
        self.By = np.zeros(n_cells)
        self.Bz = np.zeros(n_cells)

        self.rho = np.zeros(n_cells) # Charge density
        self.Jx = np.zeros(n_cells)   # Current density in x direction
        self.Jy = np.zeros(n_cells)   # Current density in y direction
        self.Jz = np.zeros(n_cells)   # Current density in z direction

    def init_fields(self, B0):
        '''initialize fields for the Harris sheet'''
        L = 0.1 # Harris sheet thickness
        x_center = self.x_max / 2.0
        B0 = 1.0 # Asymptotic magnetic field strength

        # Initialize the reversing magnetic field (e.g., in the y-direction)
        for i in range(self.nx):
            x_pos = i * self.dx
            self.By[i] = B0 * np.tanh((x_pos - x_center) / L)

    def solve_fields(self, dt):
        """Solve Maxwell's equations."""
        # Update E and B based on J and rho using a simple finite-difference time-domain (FDTD) method
        inv_dx = 1.0 / self.dx

        # 1. Update B-field (Faraday's Law)
        # Only By and Bz can change via d/dx
        for i in range(self.nx):
            next_i = (i + 1) % self.nx
            # dB_y/dt = dE_z/dx
            self.By[i] += (self.Ez[next_i] - self.Ez[i]) * (dt * inv_dx)
            # dB_z/dt = -dE_y/dx
            self.Bz[i] -= (self.Ey[next_i] - self.Ey[i]) * (dt * inv_dx)

        # 2. Update E-field (Ampere's Law)
        for i in range(self.nx):
            prev_i = (i - 1) % self.nx
            # dE_y/dt = -dB_z/dx - J_y
            curl_B_z = (self.Bz[i] - self.Bz[prev_i]) * inv_dx
            self.Ey[i] += (-curl_B_z - self.Jy[i]) * dt
            
            # dE_z/dt = dB_y/dx - J_z
            curl_B_y = (self.By[i] - self.By[prev_i]) * inv_dx
            self.Ez[i] += (curl_B_y - self.Jz[i]) * dt
            
        # Longitudinal Field (Electrostatic)
        # Gauss's Law: dE_x/dx = rho
        net_rho = np.mean(self.rho)
        self.Ex[0] = 0.0 # Boundary condition at the left edge
        for i in range(1, self.nx):
            self.Ex[i] = self.Ex[i-1] + (self.rho[i] - net_rho) * self.dx
        self.Ex -= np.mean(self.Ex) # Remove any net bias in Ex



    def interpolate_fields(self, x):
        """Interpolate E and B fields at particle position x."""

        inv_dx = 1.0 / self.dx

        relative_pos = (x - self.x_min) * inv_dx

        #set up indexing and weighting for linear interpolation
        idx_l = int(np.floor(relative_pos)) % self.nx      
        idx_r = (idx_l + 1) % self.nx
        h = relative_pos - np.floor(relative_pos) #fractional distance to the right cell
        W_l = 1 - h
        W_r = h

        #interpolate E fields
        Ex = W_l * self.Ex[idx_l] + W_r * self.Ex[idx_r]
        Ey = W_l * self.Ey[idx_l] + W_r * self.Ey[idx_r]
        Ez = W_l * self.Ez[idx_l] + W_r * self.Ez[idx_r]

        #interpolate B fields
        Bx = W_l * self.Bx[idx_l] + W_r * self.Bx[idx_r]
        By = W_l * self.By[idx_l] + W_r * self.By[idx_r]
        Bz = W_l * self.Bz[idx_l] + W_r * self.Bz[idx_r]

        return np.array([Ex, Ey, Ez]), np.array([Bx, By, Bz])
