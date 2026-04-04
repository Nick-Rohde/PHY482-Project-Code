import numpy as np

class Species:
    def __init__(self, name, mass, charge, n_particles):
        self.name = name
        self.mass = mass
        self.charge = charge
        self.n_particles = n_particles
        
        #position occurs at full steps in time
        #Velocity occurs at half steps in time
        self.x = np.zeros(n_particles)
        self.v = np.zeros(n_particles)

    def deposite_charge_currents(self, grid):
        #--- CRRENT DENSITYS ARE AT HALF STEPS IN TIME ---
        grid.rho.fill(0)
        grid.J.fill(0)

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

                grid.J[idx_l] += self.charge * self.v[i] * weights[0] * inv_dx
                grid.J[idx_r] += self.charge * self.v[i] * weights[1] *inv_dx

    def push(self, grid, dt):
        """The Boris Pusher logic."""
        for i in range(self.n_particles):
            #Get E, B field at the particle
            E,B= grid.interpolate_fields(self.x[i])

            #Boris pusher steps
            # Step 1: Half acceleration by E
            v_minus = self.v[i] + (self.charge * E / self.mass) * (dt / 2)
            
            #setp 2: Rotation by B
            t = (self.charge * B / self.mass) * (dt / 2)
            s = 2 * t / (1 + t**2)
            v_prime = v_minus + np.cross(v_minus, t)
            v_plus = v_minus + np.cross(v_prime, s)

            # Step 3: Half acceleration by E again
            self.v[i] = v_plus + (self.charge * E / self.mass) * (dt / 2)

            # Update position
            self.x[i] += self.v[i] * dt

    def apply_periodic_boundary(self, grid_length):
        """Apply periodic boundary conditions."""
        for i in range(self.n_particles):
            if self.x[i] > grid_length:
                self.x[i] = np.mod(self.x[i], grid_length)
