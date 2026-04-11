from matplotlib.pyplot import grid
import numpy as np

class Species:
    def __init__(self, name, mass, charge, n_particles):
        self.name = name
        self.mass = mass
        self.charge = charge / n_particles
        self.n_particles = n_particles
        
        #position occurs at full steps in time
        #Velocity occurs at half steps in time
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

            idx_l = idx_l % grid.nx
            idx_r = idx_r % grid.nx

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
            E,B= grid.interpolate_fields(self.x[i])

            #velocity vector
            v = np.array([self.vx[i], self.vy[i], self.vz[i]])

            #Boris pusher steps
            # Step 1: Half acceleration by E
            v_minus = v + (self.charge * E / self.mass) * (dt / 2)
            
            #setp 2: Rotation by B
            t = (self.charge * B / self.mass) * (dt / 2)
            tmag2 = np.dot(t, t)
            s = 2 * t / (1 + tmag2)
            v_prime = v_minus + np.cross(v_minus, t)
            v_plus = v_minus + np.cross(v_prime, s)

            # Step 3: Half acceleration by E again
            v_fin = v_plus + (self.charge * E / self.mass) * (dt / 2)

            self.vx[i], self.vy[i], self.vz[i] = v_fin

            # Update position
            self.x[i] += self.vx[i] * dt

            #periodic boundary conditions
            L = grid.x_max - grid.x_min
            if self.x[i] < grid.x_min:
                self.x[i] += L
            elif self.x[i] >= grid.x_max:
                self.x[i] -= L
   


    def initialize_harris_particles(self, L_sheet, n0, v_th, u_drift, grid, bg_frac = 0.2):
        """
        Initializes particle positions according to sech^2(z/L) 
        and velocities with a shifted Maxwellian (drift).
        """
        #set up fractions for background and sheet particles
        n_bg = int(self.n_particles * bg_frac)
        n_sheet = self.n_particles - n_bg

        x_center = (grid.x_max + grid.x_min) / 2.0
    
        #Set up position/velocities for sheet particles

        # Sampling from z = L * arctanh(2*rand - 1)
        P = np.random.uniform(0.001, 0.999, n_sheet) # Avoid log(0)
        x_sheet = x_center + (L_sheet / 2.0) * np.arctanh(2 * P - 1)
        x_sheet = np.clip(x_sheet, grid.x_min + 1e-6, grid.x_max - grid.dx - 1e-6)
        
        # v_x and v_z are centered at 0, v_y has the drift u_drift
        vx_sheet = np.random.normal(0, v_th, n_sheet)
        vy_sheet = np.random.normal(0, v_th, n_sheet)
        vz_sheet = np.random.normal(u_drift, v_th, n_sheet)

        # Set up position/velocities for background particles
        x_bg = np.random.uniform(grid.x_min, grid.x_max, n_bg)

        vx_bg = np.random.normal(0, v_th, n_bg)
        vy_bg = np.random.normal(0, v_th, n_bg)
        vz_bg = np.random.normal(0, v_th, n_bg)


        #array combinding
        self.x = np.concatenate([x_sheet, x_bg])
        self.vx = np.concatenate([vx_sheet, vx_bg])
        self.vy = np.concatenate([vy_sheet, vy_bg])
        self.vz = np.concatenate([vz_sheet, vz_bg])

        

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

            idx_l = idx_l % grid.nx
            idx_r = idx_r % grid.nx
            
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
