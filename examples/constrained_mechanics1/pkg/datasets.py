import torch; import numpy as np;

# more information: http://atzberger.org/

class GenDataArm1Rigid(torch.utils.data.Dataset):    
  r"""Generates samples of data."""
  
  def __init__(self,**params):
    num_samples, flag_verbose, device = tuple(map(params.get,['num_samples','flag_verbose','device']));
    num_theta, num_dim, noise_type, params_noise = tuple(map(params.get,['num_theta','num_dim','noise_type','params_noise']));
    ell_list, theta_list, X0_list, sample_mode = tuple(map(params.get,['ell_list','theta_list','X0_list','sample_mode']));
    
    if device is None:
      device = torch.device('cpu');
    
    if num_dim is None:
      num_dim = 2;    
    
    if num_theta is None:
      num_theta = 2;
    
    if ell_list is None:
      ell_list = torch.ones(num_samples,num_theta,device=device);
    
    if theta_list is None:
      theta_list = torch.rand(num_samples,num_theta,device=device)*2*np.pi;
    
    if X0_list is None:
      X0_list = torch.zeros(num_samples,num_dim,device=device);
    
    if noise_type is None:
      noise_type = 'none';
    
    if sample_mode is None:
      sample_mode = 'no_noise';
              
    # generate the arm configurations in R^4
    params_gen = {'theta_list':theta_list,'ell_list':ell_list,'X0':X0_list,'device':device};
    XX = GenDataArm1Rigid.gen_config(**params_gen);
    X = XX[:,1:3,:];
    X = X.reshape(X.shape[0],X.shape[2]*2);    
    self.params_gen = params_gen;
    
    self.noise_type = noise_type;
    self.params_noise = params_noise;

    if noise_type == 'none':
      self.samples_X_target = X;
    elif noise_type == 'Gaussian1':
      self.samples_X_target = X; 
      sigma = params_noise['sigma'];      
      eta = sigma*torch.randn(*X.shape,device=device);      
      X = X + eta;
      self.samples_X_noise = X;
    else:
      raise Exception("Not recognized, noise_type = " + noise_type);

    self.set_sample_mode(sample_mode);
    
  @staticmethod
  def create_Rotation(theta,**params):

    device, = tuple(map(params.get,['device']));
    if device is None:
      device = torch.device('cpu');

    num_dim = 2; num_samples = theta.shape[0];
    R = torch.zeros(num_samples,num_dim,num_dim,device=device);
    R[:,0,0] = torch.cos(theta); R[:,0,1] = -torch.sin(theta);
    R[:,1,0] = torch.sin(theta); R[:,1,1] = torch.cos(theta);

    return R;

  @staticmethod
  def gen_config(**params):
    theta_list,ell_list,X0_list,num_dim,device = tuple(map(params.get,['theta_list','ell_list','X0_list','num_dim','device']));
    
    if device is None:
      device = torch.device('cpu');    
    
    if num_dim is None:
      num_dim = 2;
    
    if theta_list is None:
      num_samples = int(1e3); num_theta = 2;
      theta_list = torch.rand(num_samples,num_theta,device=device)*2*np.pi;
    else:    
      num_samples = theta_list.shape[0]; num_theta = theta_list.shape[1];
    
    if ell_list is None:
      ell_list = 1.0*torch.ones(num_samples,num_theta,device=device); 

    if X0_list is None:
      X0_list = torch.zeros(num_samples,num_dim,device=device);
    
    params_R = {'device':device};
        
    X0 = X0_list;
    X = X0.unsqueeze(1); # save points  
    for k in range(0,num_theta):

      if k == 0:
        v1 = torch.zeros(num_samples,num_dim,device=device); v1[:,0] = 1.0;
        X1 = X0.clone();
      else:
        v1 = (X1 - X0); norm_v1 = torch.sqrt(torch.sum(torch.pow(v1,2),1)).unsqueeze(1); # unit vector
        v1 = v1/norm_v1;

      XX = ell_list[:,k].unsqueeze(1)*v1;
      R = GenDataArm1Rigid.create_Rotation(theta_list[:,k],**params_R);
      X2 = X1.unsqueeze(2) + torch.bmm(R,XX.unsqueeze(2)); X2 = X2.squeeze(2);

      if k == 0:
        X0 = X0; X1 = X2;
      else:
        X0 = X1; X1 = X2;

      X = torch.cat((X,X2.unsqueeze(1)),dim=1);

    return X;

  def __len__(self):    
    return self.samples_X.size()[0];

  def __getitem__(self,index):
    return self.samples_X[index];

  def to(self,device): 
    self.samples_X = self.samples_X.to(device);
        
    return self;

  def set_sample_mode(self,sample_mode):

    if sample_mode == 'no_noise':
      self.samples_X = self.samples_X_target;
    elif sample_mode == 'with_noise':
      self.samples_X = self.samples_X_noise; 
    else:
      raise Exception("Not recognized, sample_mode = " + sample_mode);

    self.sample_mode = sample_mode;
    
class GenDataArm1Klein(torch.utils.data.Dataset):    
  r"""Generates samples of data."""
  
  def __init__(self,**params):
    num_samples, flag_verbose, device = tuple(map(params.get,['num_samples','flag_verbose','device']));
    num_dim_X, noise_type, params_noise = tuple(map(params.get,['num_dim_X','noise_type','params_noise']));  # number of dimensions in which to embed
    u_list, X0_list,params_klein, sample_mode = tuple(map(params.get,['u_list','X0_list','params_klein','sample_mode']));
            
    num_dim_u = 2;
    
    if device is None:
      device = torch.device('cpu');
    
    if sample_mode is None:
      sample_mode = 'no_noise';
    
    if num_dim_X is None:
      num_dim_X = 4;

    if u_list is None:
      u_list = torch.rand(num_samples,num_dim_u,device=device);
      u_list[:,0] = 2*np.pi*u_list[:,0]; u_list[:,1] = 2*np.pi*u_list[:,1];

    if X0_list is None:
      X0_list = torch.zeros(num_samples,num_dim_X,device=device);
    
    if noise_type is None:
      noise_type = 'none';    

    if params_klein is None:
      params_klein = {}; params_klein.update({'a':1,'b':1.6,'c':0.7,'n1':50,'n2':50,'device':device});

    # generate the arm configurations in R^4    
    X = GenDataArm1Klein.func_klein_R4_1(u_list,params_klein);
    self.params_klein = params_klein;

    self.noise_type = noise_type;
    self.params_noise = params_noise;

    if noise_type == 'none':
      self.samples_X_target = X;
    elif noise_type == 'Gaussian1':
      self.samples_X_target = X; 
      sigma = params_noise['sigma'];      
      eta = sigma*torch.randn(*X.shape,device=device);      
      X = X + eta;
      self.samples_X_noise = X;
    else:
      raise Exception("Not recognized, noise_type = " + noise_type);

    self.set_sample_mode(sample_mode);

  def set_sample_mode(self,sample_mode):

    if sample_mode == 'no_noise':
      self.samples_X = self.samples_X_target;
    elif sample_mode == 'with_noise':
      self.samples_X = self.samples_X_noise; 
    else:
      raise Exception("Not recognized, sample_mode = " + sample_mode);

    self.sample_mode = sample_mode;

  @staticmethod
  def func_klein_R4_1(u,params=None):
      a,b,c,device = tuple(map(params.get,['a','b','c','device']));

      if a is None:
        a = 3;
      if b is None:
        b = 4;  
      if c is None:
        c = 2;    
      if device is None:
        device = torch.device('cpu');

      num_dim_x = 4; num_samples_u = u.shape[0];
      x = torch.zeros(num_samples_u,num_dim_x,device=device);
      u1 = u[:,0]; u2 = u[:,1];
      x[:,0] = (a + b*torch.cos(u2))*torch.cos(u1);
      x[:,1] = (a + b*torch.cos(u2))*torch.sin(u1);
      x[:,2] = b*torch.sin(u2)*torch.cos(0.5*u1);
      x[:,3] = b*torch.sin(u2)*torch.sin(0.5*u1);
    
      return x;

  def __len__(self):    
    return self.samples_X.size()[0];

  def __getitem__(self,index):
    return self.samples_X[index];

  def to(self,device): 
    self.samples_X = self.samples_X.to(device);
        
    return self;

