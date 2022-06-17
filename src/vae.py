"""
  .. image:: vae_py.png
  
  Variational Autoencoders (VAEs) for training neural networks and other models
  for learning latent structures from datasets based on Bayesian inference and 
  variational approximations.
 
  If you find these codes or methods helpful for your project, please cite our related work.

"""
# more information: http://atzberger.org/

__version__ = '1.0.0';

import torch,torch.nn as nn,numpy as np;
 
def eval_tilde_L_B(theta,phi,X_batch,**extra_params):
  r"""
  Evaluates the VAE loss function based on ELBO and the introduced regularization terms.
  
  Parameters:
    theta (model): model data structure for encoder.
    phi (model): model data structure for the decoder.
    X_batch (Tensor): collection of training data with which to compare the model results. 
    extra_params (dict): additional parameters (see examples and codes).     
    
  Returns:
    **loss** *(Tensor)* -- loss evaluation for :math:`\mathcal{L}_B` (see paper for details).

  **extra_params** [members]

  =============================  =======================================
  **Property**                   **Description**
  -----------------------------  ---------------------------------------
  m1_mc                          number of samples to use in MC estimators
  batch_size                     size of training batch
  total_N_xi                     total number of x_i's
  num_dim_x                      number of dimensions for x
  num_dim_z                      number of dimensions for z
  device                         cpu, gpu, or other device to use  
  beta                           Kullback-Leibler weight term 
                                 as in :math:`\beta`-VAEs 
  =============================  =======================================
    
  """
  X = X_batch; # to simplify notation
    
  # dereference parameters
  m1_mc      = extra_params['m1_mc']; # number of samples to use in MC estimators
  batch_size = extra_params['batch_size']; # size of training batch
  total_N_xi = extra_params['total_N_xi']; # total number of x_i's
  num_dim_x  = extra_params['num_dim_x']; # number of dimensions for x
  num_dim_z  = extra_params['num_dim_z']; # number of dimensions for z
  device     = extra_params['device']; # devide to use
  
  if 'beta' in extra_params:
    beta = extra_params['beta']; # KL weight term, as in beta-VAEs 
  else:
    beta = 1.0; # gives default value, standard VAEs
      
  return_extra_vals = None;
  if 'return_extra_vals' in extra_params:
    return_extra_vals = extra_params['return_extra_vals']; # indicates extra data to be tracked and returned
    
  model_mu_theta = theta['model_mu']; # model for the mean (neural network)
  model_log_sigma_sq_theta = theta['model_log_sigma_sq']; # model for the covariance on log scale (neural network)
    
  nu = theta['nu']; # the nu value for the latent-space prior Gaussian distribution
    
  model_mu_phi = phi['model_mu']; # model for the mean (neural network)
  model_log_sigma_sq_phi = phi['model_log_sigma_sq']; # model for the covariance on log scale (neural network)
    
  # Compute some common terms  
  mu_phi = model_mu_phi(X); # assumes tensor [batch_size,num_dim_z]
  sigma_sq_phi = torch.exp(model_log_sigma_sq_phi(X)); # assumes tensor [batch_size,num_dim_z] (diagonal of matrix)

  J1 = torch.sum(sigma_sq_phi,1) + torch.sum(mu_phi*mu_phi,1); # tensor [batch_size]
  J2 = num_dim_z;
    
  # Compute the KL-Divergence term 
  log_2_pi = np.log(2*np.pi);
  log_det_sigma_sq_phi = torch.sum(torch.log(sigma_sq_phi),1); # take log then sum.
  eps_zero = 1e-5; # WARNING: added epsilon to avoid division by zero
  D_KL_q_phi_z_x__p_theta_z  = 0.5*((1.0/(nu*nu + eps_zero))*J1 + 2*num_dim_z*torch.log(nu) + num_dim_z*log_2_pi);
  D_KL_q_phi_z_x__p_theta_z = D_KL_q_phi_z_x__p_theta_z - 0.5*(J2 + log_det_sigma_sq_phi + num_dim_z*log_2_pi);
  beta_D_KL_q_phi_z_x__p_theta_z = beta*D_KL_q_phi_z_x__p_theta_z;

  # Compute via Monte-Carlo an estimation of the expection
  epsilon_samples = torch.randn(batch_size,m1_mc,num_dim_z,device=device); # samples for Monte-Carlo estimates  
  Z               = mu_phi.unsqueeze(1) + torch.sqrt(sigma_sq_phi.unsqueeze(1))*epsilon_samples; # z ~ g_phi(epsilon,x).
  Z               = Z.reshape(batch_size*m1_mc,num_dim_z); # put into standard batch form
  mu_theta        = model_mu_theta(Z); # assumes tensor [batch_size*m1_mc,num_dim_z]
  mu_theta        = mu_theta.reshape(batch_size,m1_mc,num_dim_x); # [batch_size,m1_mc,num_dim_x]
  sigma_sq_theta  = torch.exp(model_log_sigma_sq_theta(Z)); # assumes tensor [batch_size*m1_mc,num_dim_x] (diagonal of matrix)
  sigma_sq_theta  = sigma_sq_theta.reshape(batch_size,m1_mc,num_dim_x); # [batch_size,m1_mc,num_dim_x]  
  log_det_sigma_sq_theta = torch.sum(torch.log(sigma_sq_theta),2); # [batch_size,m1_mc,1] # take log then sum
  eps_zero = 1e-5; # WARNING: added epsilon to avoid zero div.
  sigma_sq_theta_inv = 1.0/(sigma_sq_theta + eps_zero); 

  K1 = -0.5*(X.unsqueeze(1) - mu_theta)*sigma_sq_theta_inv*(X.unsqueeze(1) - mu_theta);
  K1 = torch.sum(K1,2); # sum over components of X, num_dim_x.
  K2 = -0.5*log_det_sigma_sq_theta;
  K3 = -0.5*num_dim_x*log_2_pi;
    
  avg_sum_log_p_theta__x__z = torch.sum(K1,1) + torch.sum(K2,1) + m1_mc*K3;
  avg_sum_log_p_theta__x__z = avg_sum_log_p_theta__x__z/m1_mc;

  # Compute the estimated lower bound on the log-likelihood of the dataset $\{x^{(i)}\}_{i=1}^N$  
  M = batch_size; N = total_N_xi;

  tilde_L_B_xi = -beta_D_KL_q_phi_z_x__p_theta_z + avg_sum_log_p_theta__x__z;    
  tilde_L_B    = (N/M)*torch.sum(tilde_L_B_xi,0); # sum over the batch of xi's
  
  # Return the value of $\tilde{L}^B$ (which can be backpropogated)  
  if torch.isnan(tilde_L_B):
    #print("tilde_L_B = " + str(tilde_L_B));
    #print("tilde_L_B is nan");
    raise Exception("tilde_L_B is nan.  " + "tilde_L_B = " + str(tilde_L_B));
    
  if return_extra_vals is not None: # track some of the individual term contributions
    return_extra_vals['D_KL_q_phi_z_x__p_theta_z'] = -D_KL_q_phi_z_x__p_theta_z;
    return_extra_vals['avg_sum_log_p_theta__x__z'] = avg_sum_log_p_theta__x__z;
    return_extra_vals['beta'] = beta;
      
  return tilde_L_B;

def loss_VAE_neg_tilde_L_B(X_batch,theta,phi,**extra_params):
  r"""
  Evaluates the VAE loss function based on ELBO estimator of the negative 
  log probabiliy of the data set X, loss = math:`-\tilde{\mathcal{L}}_B`

  Parameters:
    X_batch (Tensor): collection of training samples.
    theta (model): model data structure for encoder.
    phi (model): model data structure for the decoder.
    extra_params (dict): additional parameters (see examples and codes).     

  Returns:
    **loss** *(Tensor)* -- loss evaluation for :math:`-\mathcal{L}_B`.

  **extra_params** [members]

  =============================  =======================================
  **Property**                   **Description**
  -----------------------------  ---------------------------------------
  m1_mc                          number of samples to use in MC estimators
  batch_size                     size of training batch
  total_N_xi                     total number of x_i's
  num_dim_x                      number of dimensions for x
  num_dim_z                      number of dimensions for z
  device                         cpu, gpu, or other device to use  
  beta                           Kullback-Leibler weight term 
                                 as in :math:`\beta`-VAEs 
  =============================  =======================================

  """

  loss_val = -eval_tilde_L_B(theta,phi,X_batch,**extra_params);

  return loss_val;

def get_statistic_LL(theta,phi,X_batch,**extra_params):

  r"""
  We use importance sampling to estimate the Log Likelihood.
  Discussion in Burda, 2016, paper on importance weighted AE.

  :math:`LL = \log\left(p(\mathbf{x})\right)\approx\log\left(\frac{1}{m}\sum_{j = 1}^m \frac{\tilde{p}_{\theta_e,\theta_d}(\mathbf{x},\mathbf{z}^{(j)})}{\mathfrak{q}_{\theta_e}(\mathbf{z}^{(j)} | \mathbf{x})}\right).`

  The samples are taken :math:`\mathbf{z}^{(j)} \sim \mathfrak{q}_{\theta_e}(\mathbf{z}^{(j)} | \mathbf{x})`.
  Here the joint-probability under the VAE model is given by :math:`\tilde{p}_{\theta_e,\theta_d}(\mathbf{x},\mathbf{z}) = \mathfrak{p}_{\theta_d}(\mathbf{x} | \mathbf{z}) \mathfrak{p}_{\theta_d}(\mathbf{z}).`

  Parameters: 
    theta (dict): decoder model
    phi (dict): encoder model
    X_batch (Tensor) : input points.  Tensor of shape = [num_x,num_dim_x].
    extra_params (dict): extra parameters (see examples and codes).

  Returns:
    **LL** (double) -- statistic LL.

  """

  m,num_dim_z,device = tuple(map(extra_params.get,['m','num_dim_z','device']));

  if m is None:
    m = int(1e3); # number of samples to use

  if num_dim_z is None:
    raise Exception("Need to specify num_dim_z.");

  if device is None:
    device = X_batch.device;

  X = X_batch;
  num_dim_x = X.shape[1]; num_x = X.shape[0];

  # -- generate a collection of sample $z^{(j)} ~ q(z|x)$
  encoder_model_mu = phi['model_mu'];
  encoder_model_log_sigma_sq = phi['model_log_sigma_sq'];

  mu_x = encoder_model_mu(X); # shape = [num_x,num_dim_z]
  mu_x = mu_x.unsqueeze(1); # shape = [num_x,1,num_dim_z]

  sigma_sq_x = torch.exp(encoder_model_log_sigma_sq(X));
  sigma_sq_x = sigma_sq_x.unsqueeze(1); # shape = [num_x,1,num_dim_z]

  sigma_x = torch.sqrt(sigma_sq_x); # shape = [num_x,1,num_dim_z]

  eta = torch.randn(num_x,m,num_dim_z,device=device); 
  z = mu_x + sigma_x*eta; # shape [num_x,m,num_dim_z]
    
  # -- evaluate p(x|z) and p(z) to get p(x,z)
  decoder_model_mu = theta['model_mu'];
  decoder_model_log_sigma_sq = theta['model_log_sigma_sq'];

  zz = z.reshape(num_x*m,num_dim_z);
  mu_z = decoder_model_mu(zz);
  mu_z = mu_z.reshape(num_x,m,num_dim_x); # shape = [num_x,m,num_dim_x]
  sigma_sq_z = torch.exp(decoder_model_log_sigma_sq(zz));
  sigma_sq_z = sigma_sq_z.reshape(num_x,m,num_dim_x); # shape = [num_x,m,num_dim_x]
  sigma_z = torch.sqrt(sigma_sq_z); 

  eta = torch.randn(num_x,m,num_dim_x,device=device);
  x = mu_z + sigma_z*eta; # shape = [num_x,m,num_dim_x]

  nu = theta['nu'];
  p_z = torch.pow(2*np.pi*nu*nu,-num_dim_z/2.0)*torch.exp(-torch.sum(torch.pow(z,2)/(2.0*nu*nu),2)); # Gaussian density prior, shape = [num_x,m]
  p_x_given_z = torch.pow(torch.prod(2*np.pi*sigma_sq_z,2),-1.0/2.0)*torch.exp(-torch.sum(torch.pow(x - mu_z,2)/(2.0*sigma_sq_z),2)); # Gaussian density, shape = [num_x,m]

  p_x_z = p_x_given_z*p_z;

  # -- evalue q(z|x) 
  q_z_given_x = torch.pow(torch.prod(2*np.pi*sigma_sq_x,2),-1.0/2.0)*torch.exp(-torch.sum(torch.pow(z - mu_x,2)/(2.0*sigma_sq_x),2)); # Gaussian density, shape = [num_x,m]

  # -- compute approximation to expectation E[p(x,z)/q(z|x)]
  # where z ~ q(z|x)
  R = p_x_z/q_z_given_x;
  exp_R = torch.sum(R,1)/m;
  log_exp_R = torch.log(exp_R);

  return log_exp_R;
   
def get_statistic_KL(theta,phi,X_batch,**extra_params):
  r"""Compute estimate of the Kullback-Leibler divergence 
      :math:`D_KL(q(z|x) | p(z)) = E_q[log(q(z|x))] - E_q[log(p(z))]` 
  """

  m,num_dim_z,device = tuple(map(extra_params.get,['m','num_dim_z','device']));

  if m is None:
    m = int(1e3); # number of samples to use

  if num_dim_z is None:
    raise Exception("Need to specify num_dim_z.");

  if device is None:
    device = X_batch.device;

  X = X_batch;
  num_dim_x = X.shape[1]; num_x = X.shape[0];

  # -- generate a collection of sample $z^{(j)} ~ q(z|x)$
  encoder_model_mu = phi['model_mu'];
  encoder_model_log_sigma_sq = phi['model_log_sigma_sq'];

  mu_x = encoder_model_mu(X_batch);
  mu_x = mu_x.unsqueeze(1);

  sigma_sq_x = torch.exp(encoder_model_log_sigma_sq(X_batch));
  sigma_sq_x = sigma_sq_x.unsqueeze(1);

  sigma_x = torch.sqrt(sigma_sq_x); # shape = [num_x,1,num_dim_z]

  eta = torch.randn(num_x,m,num_dim_z,device=device);
  z = mu_x + sigma_x*eta; # shape = [num_x,m,num_dim_z]

  # -- D_KL(q | p) = E_q[log(q(z|x))] - E_q[log(p(z))]
  nu = theta['nu'];
  p_z = torch.pow(2*np.pi*nu*nu,-num_dim_z/2.0)*torch.exp(-torch.sum(torch.pow(z,2)/(2.0*nu*nu),2)); # Gaussian density prior, shape = [num_x,m]

  # evaluate q(z|x) 
  q_z_given_x = torch.pow(torch.prod(2*np.pi*sigma_sq_x,2),-1.0/2.0)*torch.exp(-torch.sum(torch.pow(z - mu_x,2)/(2.0*sigma_sq_x),2)); # Gaussian density, shape = [num_x,m]

  E1 = torch.sum(torch.log(q_z_given_x),1)/m;
  E2 = torch.sum(torch.log(p_z),1)/m;

  D_KL = E1 - E2;

  return D_KL;
   
def get_statistic_RE(theta,phi,X_batch,**extra_params):
  r""" Compute estimate of the reconstruction error term
  :math:`RE = E_q[log(p(x|z))] - E_q[log(p(z))]` """

  m,num_dim_z,device = tuple(map(extra_params.get,['m','num_dim_z','device']));

  if m is None:
    m = int(1e3); # number of samples to use

  if num_dim_z is None:
    raise Exception("Need to specify num_dim_z.");

  if device is None:
    device = X_batch.device;

  X = X_batch;
  num_dim_x = X.shape[1]; num_x = X.shape[0];

  # -- generate a collection of sample $z^{(j)} ~ q(z|x)$
  encoder_model_mu = phi['model_mu'];
  encoder_model_log_sigma_sq = phi['model_log_sigma_sq'];

  mu_x = encoder_model_mu(X_batch);
  mu_x = mu_x.unsqueeze(1);

  sigma_sq_x = torch.exp(encoder_model_log_sigma_sq(X_batch));
  sigma_sq_x = sigma_sq_x.unsqueeze(1);

  sigma_x = torch.sqrt(sigma_sq_x); # shape = [num_x,1,num_dim_z]

  eta = torch.randn(num_x,m,num_dim_z,device=device);
  z = mu_x + sigma_x*eta;

  decoder_model_mu = theta['model_mu'];
  decoder_model_log_sigma_sq = theta['model_log_sigma_sq'];

  zz = z.reshape(num_x*m,num_dim_z);
  mu_z = decoder_model_mu(zz);
  mu_z = mu_z.reshape(num_x,m,num_dim_x); # shape = [num_x,m,num_dim_x]
  sigma_sq_z = torch.exp(decoder_model_log_sigma_sq(zz));
  sigma_sq_z = sigma_sq_z.reshape(num_x,m,num_dim_x); # shape = [num_x,m,num_dim_x]
  sigma_z = torch.sqrt(sigma_sq_z); 

  eta = torch.randn(num_x,m,num_dim_x,device=device);
  x = mu_z + sigma_z*eta; # shape = [num_x,m,num_dim_x]

  # -- RE
  p_x_given_z = torch.pow(torch.prod(2*np.pi*sigma_sq_z,2),-1.0/2.0)*torch.exp(-torch.sum(torch.pow(x - mu_z,2)/(2.0*sigma_sq_z),2)); # Gaussian density, shape = [num_dim,m]

  E1 = torch.sum(p_x_given_z,1)/m;
 
  return E1;
  
def get_statistic_ELBO(theta,phi,X_batch,**extra_params):
  r"""Returns the Evidence Lower Bound (ELBO) for models.
     The :math:`ELBO = -\mathcal{L}_B`.
  """
  L_B = eval_tilde_L_B(theta,phi,X_batch,**extra_params);
  return -L_B;

