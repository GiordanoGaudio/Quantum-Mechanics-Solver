import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

class QM_System:
    """
    
    This solver is designed to handle 1D QM Time Independent Systems
    under the assumed boundary condition
    
    ψ(0) = ψ(1) = 0
    
    with the assumption that energies are in units of 2m(dx/hbar)**2
    
    """
    
    
    
    def __init__(self, V, N = 150, x_min = 0, x_max = 1, k = 10):
        
        # Basic properties
        
        self.V = V
        self.N = N
        self.k = k
        self.x_min = x_min
        self.x_max = x_max
        
        # Geometric properties
        
        self.L = x_max - x_min                          # Length
        self.dx = self.L/N                              # Step size
        self.x_ax = np.arange(self.x_min + self.dx,     # x-axis
                              self.x_max + self.dx, 
                              self.dx)
        
        # Potential Energy
        
        self.V_vec  = self.V(self.x_ax)
        self.U = sparse.diags(self.V_vec, (0))
        
        # Kinetic Energy
        
        self.Nones = np.ones(self.N)
        self.diags =  np.array([self.Nones, -2*self.Nones, self.Nones])
        self.Laplacian = sparse.spdiags(self.diags, np.array([-1,0,1]), self.N, self.N)
        self.T = -self.Laplacian/2
        
        # Hamiltonian
        
        self.H = self.T + self.U
        
        
    def Eig(self):
        values, vectors = eigsh(self.H, k = self.k, which = 'SM')
        return [values, vectors]
    
    def E(self, n):
        E = self.Eig()[0]  # the first k energies
        return E.T[n]
    
    def ψ(self, n):
        ψ = self.Eig()[1] # the first k wavefunctions
        ψ = ψ.T[n]
        ψ = ψ/(np.linalg.norm(ψ)) # normalize
        return ψ
  