# general packages
import sys,os,numpy as np;
import torch;

# local packages
from . import model_utils;
BurgersPdeSolver = model_utils.BurgersPdeSolver;

# --
class PeriodicDataset(torch.utils.data.Dataset):
  r"""
    Creates a periodic dataset of functions evolving under Burgers' PDE.  
    At t = 0, this has the topology of a periodic.
  """
  
  def __init__(self, xi, num_samples, params):
    r"""    
    Parameters:
      xi: 
      num_samples: number of samples to generate
      params (dict): additional parameters
        
    params:
        time_step (double): the time-step to use
        noise (double): adds Gaussian noise to input and target samples
        nu (double): latent space variance for prior                                
    """
    # angles (Tensor): specifies latent variables to generate data (optional)
    # times (Tensor): specifies latent variables to generate data (optional)
    
    super().__init__();
    
    self.xi = xi;
    self.num_samples = num_samples;
    self.time_step = params.get('time_step', 0);
    self.nu = params.get('nu', 0.02); # nu for Burgers Equation
    self.burgers_pde_solver = BurgersPdeSolver(len(xi), nu=self.nu, time_step=self.time_step);
    self.noise = params.get('noise', 0);
    angles = 2*np.pi*torch.rand(num_samples);
    times = torch.rand(num_samples,1);
    self.samples_X = self.create_data(angles, times);
    self.samples_Y = self.create_data(angles, times+self.time_step);
    
    if self.noise != 0:
        self.samples_X += torch.normal(0.0, self.noise, (self.num_samples, len(self.xi)));
        self.samples_Y += torch.normal(0.0, self.noise, (self.num_samples, len(self.xi)));
        
    self.times = times;

  def __len__(self):
    return self.samples_X.shape[0];

  def __getitem__(self,index):
    return self.samples_X[index], self.samples_Y[index];
  
  def _get_basis_vectors(self):
    r"""
    Returns two orthonormal basis vectors with same dimension as input space.
    """
    v1 = torch.cos(2*np.pi*self.xi); v2 = torch.sin(2*np.pi*self.xi);
    return v1, v2;

  def _get_cartesian_coords(self, angles):
    z1 = torch.cos(angles); z2 = torch.sin(angles);
    return z1, z2;

  def create_data(self, angles, times): 
    r"""
    Creates data according to the function:

    u(x,t=0;angle)=[cos(angle)cos(2pi*x)+sin(angle)sin(2pi*x)]
    """
    v1, v2 = self._get_basis_vectors(); z1, z2 = self._get_cartesian_coords(angles);
    data = 0.5*(torch.outer(z1,v1)+torch.outer(z2,v2));
    data = self.burgers_pde_solver.evolve_burgers(data, time_step = times.flatten().numpy());    
    return data;

  
  
