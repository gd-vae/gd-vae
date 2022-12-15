import torch,numpy as np;

# setup surface description for sigma(u)
def get_manifold_sigma_info_klein1(x,params=None):
  # assumes x is 4D point on Klein bottle
  # chooses local coordinate chart 
  #   chart 1: sigma(u) = [x1,x2,x3,x4] (see below)
  #
  # \begin{eqnarray}
  #  x_1 & = & (a + b\cos(u_2))\cos(u_1) \\
  #  x_2 & = & (a + b\cos(u_2))\sin(u_1) \\
  #  x_3 & = & b\sin(u_2)\cos\left(\frac{u_1}{2}\right) \\
  #  x_4 & = & b\sin(u_2)\sin\left(\frac{u_1}{2}\right).
  # \end{eqnarray}
  #
  #device = params['device'];  
  params_klein = params['params_klein'];
  device = x.device;

  results = {}; num_samples = x.shape[0]; num_dim_x = x.shape[1]; num_dim_u = 2;

  # get chart and point information
  manifold_chart_I,manifold_chart_u,manifold_ptsX = tuple(map(params.get,['manifold_chart_I','manifold_chart_u',
                                                                          'manifold_ptsX']));

  I_x = params['I_x']; # assumes set already by recent closest point algorithm call for same x
  chart_I = manifold_chart_I[I_x]; u = manifold_chart_u[I_x,:];
  xx = manifold_ptsX[I_x,:]; # makes consistent

  # chart 0: # prototype for more general case (assumes one chart for now)
  II = torch.nonzero(chart_I == 0).squeeze(1);
  if II.shape[0] != num_samples:
    raise Exception("Assuming only one coordiate chart used for Klein Bottle currently.");      
  # We can use automatic differentiaion to get the derivatives needed and values
  # (we recompute the surface points to build local comp. call graph for taking derivatives)
  # this yields the 
  #   sigma_k with shape = [num_samples_u,num_dim_x]
  #   d_ui_sigma_k[II,i,k]  with shape = [num_samples_u,num_dim_u,num_dim_x]
  #   d_ui_uj_sigma_k with shape = [num_samples_u,num_dim_u,num_dim_u,num_dim_x]  
  with torch.enable_grad():
    sigma_k = torch.zeros(num_samples,num_dim_x,device=device);
    d_ui_sigma_k = torch.zeros(num_samples,num_dim_u,num_dim_x,device=device);
    d_ui_uj_sigma_k = torch.zeros(num_samples,num_dim_u,num_dim_u,num_dim_x,device=device);

    uu = u[II,:]; # get all u for the current chart
    uu = uu.detach().requires_grad_(True); # detach to make leaf variable of local comp. graph    
    sigma_k[II,:] = func_klein_R4_1(uu,params=params_klein); # sigma(u^*) = x (by assumption)    
    for k in range(0,num_dim_x):
      ss = torch.sum(sigma_k[II,k]);
      d_u_ss, = torch.autograd.grad(ss,uu,retain_graph=True,allow_unused=True,create_graph=True);
      for i in range(0,num_dim_u):
        d_ui_sigma_k[II,i,k] = d_u_ss[:,i];
        sss = torch.sum(d_u_ss[:,i]);
        d_u_u_ss, = torch.autograd.grad(sss,uu,retain_graph=True,allow_unused=False,create_graph=True); # @@@ check no side-effects
        for j in range(0,num_dim_u): # @@@ double-check no side-effect of calling twice
          d_ui_uj_sigma_k[II,i,j,k] = d_u_u_ss[:,j];

  results.update({'sigma_k':sigma_k,'d_ui_sigma_k':d_ui_sigma_k,'d_ui_uj_sigma_k':d_ui_uj_sigma_k});

  return results;

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
        
def sample_klein_bottle_points_R4(params=None):
  
  a,b,n1,n2,device = tuple(map(params.get,['a','b','n1','n2','device']));
      
  if a is None:
    a = 3;
  if b is None:
    b = 4;  
      
  if n1 is None:
    n1 = 10; 
  if n2 is None:
    n2 = n1; 
    
  if device is None:
    device = torch.device('cpu');    
    
  num_dim_x = 4; num_samples_u = n1*n2;    
  pts_u1 = torch.linspace(0,2*np.pi,n1,device=device);
  pts_u2 = torch.linspace(0,2*np.pi,n2,device=device);
  u1,u2 = torch.meshgrid(pts_u1,pts_u2,indexing='xy');
  u1 = u1.flatten(); u2 = u2.flatten();
  u = torch.stack((u1,u2),dim=1);

  u = u.requires_grad_(True); # needed for sigma info. later 
  
  params_k = {'a':a,'b':b,'device':device};
  x = func_klein_R4_1(u,params=params_k);
    
  return x, u;

def sample_klein_bottle_points_R3(params=None):  
  device = params['device']; 

  a,b,c,n1,n2 = tuple(map(params.get,['a','b','c','n1','n2']));

  if a is None:
    a = 6;
  if b is None:
    b = 16;  
  if c is None:
    c = 2;
  if n1 is None:
    n1 = 10; 
  if n2 is None:
    n2 = n1;     
    
  num_dim_x = 3;
  num_samples_u = n1*n2;
    
  # part 1
  # see parameterization in G. Franzoni, AMS Notices, 2012,
  # Dickson Bottle shape is:
  pts_u1 = torch.linspace(0,np.pi,n1,device=device);
  pts_u2 = torch.linspace(0,2*np.pi,n2,device=device);
  u1,u2 = torch.meshgrid(pts_u1,pts_u2,indexing='xy');
  u1 = u1.flatten(); u2 = u2.flatten();

  x = torch.zeros(num_samples_u,num_dim_x,device=device);  
  r_u1 = c*(1 - 0.5*torch.cos(u1));    
  x[:,0] = (a*(1 + torch.sin(u1)) + r_u1*torch.cos(u2))*torch.cos(u1); 
  x[:,1] = (b + r_u1*torch.cos(u2))*torch.sin(u1);
  x[:,2] = r_u1*torch.sin(u2);

  u = torch.stack((u1,u2),dim=1);
  xA = x; uA = u;
    
  # part 2
  pts_u1 = torch.linspace(np.pi,2*np.pi,n1,device=device);
  pts_u2 = torch.linspace(0,2*np.pi,n2,device=device);
  u1,u2 = torch.meshgrid(pts_u1,pts_u2,indexing='xy');
  u1 = u1.flatten(); u2 = u2.flatten();

  x = torch.zeros(num_samples_u,num_dim_x,device=device);  
  r_u1 = c*(1 - 0.5*torch.cos(u1));
  x[:,0] = a*(1 + torch.sin(u1))*torch.cos(u1) + r_u1*torch.cos(u2 + np.pi);
  x[:,1] = b*torch.sin(u1);
  x[:,2] = r_u1*torch.sin(u2);
  u = torch.stack((u1,u2),dim=1);
  xB = x; uB = u;

  x = torch.cat((xA,xB),dim=0);
  u = torch.cat((uA,uB),dim=0);
 
  return x,u;
