from matplotlib.pyplot import grid
import numpy as np

class Species:
    def __init__(self, name, mass, charge, n_particles):
        self.name = name
        self.mass = mass
        self.charge = charge / n_particles
        self.n_particles = n_particles

        self.x = np.zeros(n_particles)
        self.vx = np.zeros(self.n_particles)
        self.vy = np.zeros(self.n_particles)
        self.vz = np.zeros(self.n_particles)

    def deposite_charge_currents(self, grid):
        #--- CRRENT DENSITYS ARE AT HALF STEPS IN TIME ---
        grid.rho.fill(0)
        grid.Jx.fill(0)
        grid.Jy.fill(0)

        inv_dx = 1.0 / grid.dx
        for i in range(self.n_particles):
            pos_in_cell = self.x[i] * inv_dx   #get position in terms of cell units
            idx_l = np.floor(pos_in_cell).astype(int)  #get the left cell index
            idx_r = idx_l + 1 #right cell index
            h = pos_in_cell - idx_l    #get the fractional distance to the right cell
            W_l = 1 - h #weight for left cell
            W_r = h #weight for right cell
            weights = [W_l, W_r]

            if 0 <= idx_l < grid.nx - 1:
                #update the charge density/current density on the grid using the weights
                grid.rho[idx_l] += self.charge * weights[0] *inv_dx
                grid.rho[idx_r] += self.charge * weights[1] *inv_dx

                grid.Jx[idx_l] += self.charge * self.vx[i] * weights[0] * inv_dx
                grid.Jx[idx_r] += self.charge * self.vx[i] * weights[1] *inv_dx

                grid.Jy[idx_l] += self.charge * self.vy[i] * weights[0] * inv_dx
                grid.Jy[idx_r] += self.charge * self.vy[i] * weights[1] * inv_dx



    def push(self, grid, dt):
        """The Boris Pusher logic."""
        for i in range(self.n_particles):
            #Get E, B field at the particle
            cell_index = int(self.x[i] / grid.dx)
            E,B= grid.interpolate_fields(self.x[i])

            #velocity vector
            v = np.array([self.vx[i], self.vy[i], self.vz[i]])

            #Boris pusher steps
            # Step 1: Half acceleration by E
            v_minus = v + (self.charge * E / self.mass) * (dt / 2)
            
            #setp 2: Rotation by B
            t = (self.charge * B / self.mass) * (dt / 2)
            s = 2 * t / (1 + t**2)
            v_prime = v_minus + np.cross(v_minus, t)
            v_plus = v_minus + np.cross(v_prime, s)

            # Step 3: Half acceleration by E again
            v_fin = v_plus + (self.charge * E / self.mass) * (dt / 2)

            self.vx[i], self.vy[i], self.vz[i] = v_fin

            # Update position
            self.x[i] += self.vx[i] * dt

            #reflection boundary conditions
            eps = 1e-6 * grid.dx
            if self.x[i] < grid.x_min:
                self.x[i] = grid.x_min + (grid.x_min - self.x[i]) + eps
                self.vx[i] *= -1.0
            elif self.x[i] >= grid.x_max:
                self.x[i] = grid.x_max - (self.x[i] - grid.x_max) - eps
                self.vx[i] *= -1.0
   


    def initialize_harris_particles(self, L, n0, v_th, u_drift, grid):
        """
        Initializes particle positions according to sech^2(z/L) 
        and velocities with a shifted Maxwellian (drift).
        """
        # 1. Position: Sampling from sech^2 is tricky. 
        # Usually done via inverse transform sampling: z = L * arctanh(2*rand - 1)
        P = np.random.uniform(0.001, 0.999, self.n_particles) # Avoid log(0)
        self.x = L * np.arctanh(2 * P - 1)

        self.x = np.clip(self.x, grid.x_min + 1e-6, grid.x_max - grid.dx - 1e-6)
        
        # 2. Velocity: Shifted Maxwellian
        # v_x and v_z are centered at 0, v_y has the drift u_drift
        self.vx = np.random.normal(0, v_th, self.n_particles)
        self.vy = np.random.normal(0, v_th, self.n_particles)
        self.vz = np.random.normal(u_drift, v_th, self.n_particles)

        

    def deposit_sources(self, grid):
        inv_dx = 1.0 / grid.dx
        grid.rho.fill(0)
        grid.Jy.fill(0)
        grid.Jx.fill(0)
        grid.Jz.fill(0)

        for i in range(self.n_particles):
            x_norm = (self.x[i] - (grid.x_max / 2)) * inv_dx
            idx_l = int(np.floor(x_norm))
            idx_r = idx_l + 1
            
            W_r = x_norm - idx_l
            W_l = 1.0 - W_r
            
            if 0 <= idx_l < grid.nx - 1:
                # Charge
                grid.rho[idx_l] += self.charge * W_l * inv_dx
                grid.rho[idx_r] += self.charge * W_r * inv_dx
                
                # Current in Y (the main Harris current)
                grid.Jz[idx_l] += self.charge * self.vy[i] * W_l * inv_dx
                grid.Jz[idx_r] += self.charge * self.vy[i] * W_r * inv_dx

                # Current in X (important for waves/instabilities)
                grid.Jx[idx_l] += self.charge * self.vx[i] * W_l * inv_dx
                grid.Jy[idx_l] += self.charge * self.vy[i] * W_l * inv_dx
