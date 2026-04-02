import numpy as np

class Species:
    def __init__(self, name, mass, charge, n_particles):
        self.name = name
        self.mass = mass
        self.charge = charge
        self.n_particles = n_particles

        self.x = np.zeros(n_particles)
        self.v = np.zeros(n_particles)

    def deposite_current(self, grid):
        # Placeholder for current deposition logic
        grid.rho.fill(0) # Reset charge density
        grid.J.fill(0)   # Reset current density
        for i in range(self.n_particles):
            # Simple 'cloud-in-cell' deposition scheme for 1D
            cell_index = int(self.x[i] / grid.dx)
            h = cell_index -i 
            W_r = 1 - abs(h)
            W_l = abs(h)
            weights = [W_l, W_r]
            np.add.at(grid.rho, [cell_index, cell_index+1], self.charge * weights)
            np.add.at(grid.J, [cell_index, cell_index+1], self.charge * self.v[i] * weights)

    def push(self, grid, dt):
        """The Boris Pusher logic."""
        for i in range(self.n_particles):
            #Get E, B field at the particle
            cell_index = int(self.x[i] / grid.dx)
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