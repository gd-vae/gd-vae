import torch; import numpy as np; import pickle,os,sys;
import gd_vae_pytorch as gd_vae,gd_vae_pytorch.geo_map,gd_vae_pytorch.vae;
import gd_vae_pytorch.nn,gd_vae_pytorch.utils; 
from . import geometry as pkg_geometry;

# more information: http://atzberger.org/

def create_model_gd_vae_nn1(**params):
  params_theta, params_phi, device = tuple(map(params.get,['params_theta','params_phi', 'device']));

  # -- theta (decoder)
  layer_sizes, layer_biases, manifold_map = tuple(map(params_theta.get,['layer_sizes', 'layer_biases', 
                                                                        'manifold_map']));

  ss = layer_sizes; bb = layer_biases;
  seq_list = []; num_layers = len(layer_sizes) - 1;
  for k in range(0,num_layers):
    seq_list.append(torch.nn.Linear(ss[k],ss[k+1],bias=bb[k]));
    if k != (num_layers - 1):
      seq_list.append(torch.nn.LeakyReLU(negative_slope=1e-6));

  if manifold_map is not None:
    seq_list.append(manifold_map);

  model_mu = torch.nn.Sequential(*seq_list).to(device);

  model_log_sigma_sq = torch.nn.Sequential(
    gd_vae.nn.AtzLearnableTensorLayer(tensor_size=(1,layer_sizes[-1]),device=device),    
  ).to(device);

  theta = {};
  nu = torch.Tensor([0.5]).to(device);    
  theta.update({'model_mu':model_mu,'model_log_sigma_sq':model_log_sigma_sq,'nu':nu});

  # -- phi (encoder)  
  layer_sizes, layer_biases, manifold_map = tuple(map(params_phi.get,['layer_sizes', 'layer_biases', 
                                                                      'manifold_map']));

  ss = layer_sizes; bb = layer_biases;
  seq_list = []; num_layers = len(layer_sizes) - 1;
  for k in range(0,num_layers):
    seq_list.append(torch.nn.Linear(ss[k],ss[k+1],bias=bb[k]));
    if k != num_layers - 1:
      seq_list.append(torch.nn.LeakyReLU(negative_slope=1e-6));

  if manifold_map is not None:
    seq_list.append(manifold_map);

  model_mu = torch.nn.Sequential(*seq_list).to(device);    

  model_log_sigma_sq = torch.nn.Sequential(
    gd_vae.nn.AtzLearnableTensorLayer(tensor_size=(1,layer_sizes[-1]),device=device),
  ).to(device);

  phi = {}; 
  phi.update({'model_mu':model_mu,'model_log_sigma_sq':model_log_sigma_sq});

  model = {'theta':theta,'phi':phi};

  return model;

def create_model(**params_create):    
  type_model = params_create['type_model']; 

  if type_model == 'gd_vae_nn1':
    model = create_model_gd_vae_nn1(**params_create);
  else:
    raise Exception("Not recognized, type_model = " + type_model);  

  return model;

# Get the learnable data fields from the models for initialization and use
def get_data_of_model_select_arm1(theta,phi): 
  d = {};
  
  theta_model_mu, theta_model_log_sigma_sq = tuple(map(theta.get,['model_mu','model_log_sigma_sq']));
  phi_model_mu, phi_model_log_sigma_sq = tuple(map(phi.get,['model_mu','model_log_sigma_sq']));
    
  d['nu_theta'] = theta['nu'];

  mm = theta_model_mu;
  ll = list(mm[0].parameters());
  d['mu_theta_W1'] = ll[0].data;
  if len(ll) > 1:
    d['mu_theta_b1'] = ll[1].data;
  ll = list(mm[2].parameters());
  d['mu_theta_W2'] = ll[0].data;
  if len(ll) > 1:
    d['mu_theta_b2'] = ll[1].data;
  
  if theta_model_log_sigma_sq is not None:
    qq = theta_model_log_sigma_sq[0];
    if isinstance(qq,gd_vae.nn.AtzLearnableTensorLayer):
      d['log_sigma_sq_theta_tensor'] = qq.params['tensor'];
        
  mm = phi_model_mu;        
  ll = list(mm[0].parameters());
  d['mu_phi_W1'] = ll[0].data;
  if len(ll) > 1:
    d['mu_phi_b1'] = ll[1].data;
  ll = list(mm[2].parameters());
  d['mu_phi_W2'] = ll[0].data;
  if len(ll) > 1:    
    d['mu_phi_b2'] = ll[1].data;

  if phi_model_log_sigma_sq is not None:
    qq = phi_model_log_sigma_sq[0];
    if isinstance(qq,gd_vae.nn.AtzLearnableTensorLayer):
      d['log_sigma_sq_phi_tensor'] = qq.params['tensor'];
    
  return d;

def create_manifold_map_klein1(**extra_params):
  params_klein,device = tuple(map(extra_params.get,['params_klein','device']));

  # setup manifold description as point cloud
  x,u = pkg_geometry.sample_klein_bottle_points_R4(params_klein);
  num_samples = x.shape[0]; num_dim_x = x.shape[1]; num_dim_u = u.shape[1];
  manifold_chart_I = torch.zeros(num_samples,device=device);
  manifold_chart_u = torch.zeros(num_samples,num_dim_u,device=device);
  manifold_ptsX = torch.zeros(num_samples,num_dim_x,device=device);

  # chart 0: (only one chart for now)
  chart_I = 0; I = torch.arange(0,num_samples);
  manifold_chart_I[I] = chart_I;
  manifold_chart_u[I,:] = u[I,:];
  manifold_ptsX[I,:] = x[I,:];

  # setup closest point
  params_map = {};
  params_map.update({'manifold_chart_I':manifold_chart_I,
                     'manifold_chart_u':manifold_chart_u,
                     'manifold_ptsX':manifold_ptsX});

  params_map.update({'find_nearest_manifold_pt':gd_vae.geo_map.PointCloudMapFunc.find_nearest_manifold_pt_kdtree});
  params_map.update({'find_nearest_manifold_pt_params':{'manifold_ptsX':manifold_ptsX}});
                     
  params_map.update({'get_manifold_sigma_info':pkg_geometry.get_manifold_sigma_info_klein1});
  params_map.update({'get_manifold_sigma_info_params':{'manifold_chart_I':manifold_chart_I,
                                                       'manifold_chart_u':manifold_chart_u,
                                                       'manifold_ptsX':manifold_ptsX,
                                                       'params_klein':params_klein,
                                                       'device':device}});

  manifold_map = gd_vae.geo_map.ManifoldPointCloudLayer(params_map); # can be used as part of PyTorch pipeline with backprop
  
  return manifold_map;

def create_manifold_map_torus(**extra_params):
  device, = tuple(map(extra_params.get,['device']));
  
  # setup closest point
  params_map = {'func_map':gd_vae.geo_map.map_clifford_torus,
                'func_map_params':{'num_circles':2},
                'device':device};
  
  manifold_map = gd_vae.geo_map.ManifoldDirectMapLayer(params_map); # can be used as part of PyTorch pipeline with backprop
  
  return manifold_map;

def create_manifold_map_sphere(**extra_params):
  device, = tuple(map(extra_params.get,['device']));
  
  # setup closest point  
  params_map = {'func_map':gd_vae.geo_map.map_sphere,
                'func_map_params':{'sphere_r':1.0},
                'device':device};
  
  manifold_map = gd_vae.geo_map.ManifoldDirectMapLayer(params_map); # can be used as part of PyTorch pipeline with backprop
  
  return manifold_map;

def get_data_of_model_select(theta,phi,flag_model_type):
  if flag_model_type == 'arm1':
    return get_data_of_model_select_arm1(theta,phi);        
  else:
    raise Exception('not recognized.');
    
def print_params_select_arm1(theta,phi,flag_model_type):
    # Get the data from the model and put into d
    d = gd_vae.utils.DictAsMembers(get_data_of_model_select(theta,phi,flag_model_type));        

    print("mu_theta_W1.shape = " + str(d.mu_theta_W1.shape));
    print("mu_theta_b1.shape = " + str(d.mu_theta_b1.shape));
    
    print("mu_theta_W2.shape = " + str(d.mu_theta_W2.shape));
        
    print("log_sigma_sq_theta_tensor.shape = " + str(d.log_sigma_sq_theta_tensor.shape));
                    
    print("mu_phi_W1.shape = " + str(d.mu_phi_W1.shape));
    print("mu_phi_b1.shape = " + str(d.mu_phi_b1.shape));

    print("mu_phi_W2.shape = " + str(d.mu_phi_W2.shape));
        
    print("log_sigma_sq_phi_tensor.shape = " + str(d.log_sigma_sq_phi_tensor.shape));

    print("nu_theta = " + str(d.nu_theta));

def print_params_select(theta,phi,flag_model_type):    
  if flag_model_type == 'arm1':
    print_params_select_arm1(theta,phi,flag_model_type);
  else:
    raise Exception("not recongnized.");
      
def save_model(theta,phi,type_model,filename_base,label):
  model_mu_data = {'state_dict':theta['model_mu'].state_dict(),
                   'str':str(theta['model_mu'])};
  model_log_sigma_sq_data = {'state_dict':theta['model_log_sigma_sq'].state_dict(),
                             'str':str(theta['model_log_sigma_sq'])};    
  s = {'pytorch_version':torch.__version__,
       'type_model':type_model,
       'model_mu_data':model_mu_data,
       'model_log_sigma_sq_data':model_log_sigma_sq_data};
  model_filename = '%s_theta_%s.pickle'%(filename_base,label);
  print("Saving network to model_filename = %s"%model_filename);
  pickle.dump(s, open(model_filename,"wb"));
  
  model_mu_data = {'state_dict':phi['model_mu'].state_dict(),
                             'str':str(phi['model_mu'])};
  model_log_sigma_sq_data = {'state_dict':phi['model_log_sigma_sq'].state_dict(),
                             'str':str(phi['model_log_sigma_sq'])};    
  s = {'pytorch_version':torch.__version__,
       'type_model':type_model,
       'model_mu_data':model_mu_data,
       'model_log_sigma_sq_data':model_log_sigma_sq_data};
  model_filename = '%s_phi_%s.pickle'%(filename_base,label);
  print("Saving network to model_filename = %s"%model_filename);
  pickle.dump(s, open(model_filename,"wb"));

def load_model(theta,phi,filename_base,label):    
  # Load the Model from datafile
  model_filename = '%s_theta_%s.pickle'%(filename_base,label);
  fid = open(model_filename,'rb');
  theta_data = pickle.load(fid);
  fid.close();

  model_mu = theta['model_mu'];
  model_mu.load_state_dict(theta_data['model_mu_data']['state_dict']);
  model_log_sigma_sq = theta['model_log_sigma_sq'];
  model_log_sigma_sq.load_state_dict(theta_data['model_log_sigma_sq_data']['state_dict']);

  # Load the Model from datafile
  model_filename = '%s_phi_%s.pickle'%(filename_base,label);
  fid = open(model_filename,'rb');
  phi_data = pickle.load(fid);
  fid.close();

  model_mu = phi['model_mu'];
  model_mu.load_state_dict(phi_data['model_mu_data']['state_dict']);
  model_log_sigma_sq = phi['model_log_sigma_sq'];
  model_log_sigma_sq.load_state_dict(phi_data['model_log_sigma_sq_data']['state_dict']);      
      
def init_model_arm1_pretrain_decoder1_init(theta,phi):
    flag_model_type = 'arm1';

    # Get the data from the model and put into d
    d = gd_vae.utils.DictAsMembers(get_data_of_model_select(theta,phi,flag_model_type));

    urange_w_a = urange_b_a = -2e-1; urange_w_b = urange_b_b = 2e-1;
    d.mu_theta_W1.uniform_(urange_w_a,urange_w_b);
    d.mu_theta_b1.uniform_(urange_b_a,urange_b_b);

    urange_w_a = urange_b_a = -2e-1; urange_w_b = urange_b_b = 2e-1;
    d.mu_theta_W2.uniform_(urange_w_a,urange_w_b);
    
    # Note: the [:] is important to get in-place value assignment (no history)    
    if theta['model_log_sigma_sq'] is not None:    
      d.log_sigma_sq_theta_tensor[:] = -5;

    urange_w_a = urange_b_a = -3e-1; urange_w_b = urange_b_b = 3e-1;
    d.mu_phi_W1.uniform_(urange_w_a,urange_w_b);
    d.mu_phi_b1.uniform_(urange_b_a,urange_b_b);

    urange_w_a = urange_b_a = -3e-1; urange_w_b = urange_b_b = 3e-1;
    d.mu_phi_W2.uniform_(urange_w_a,urange_w_b);
    #d.mu_phi_b2.uniform_(urange_b_a,urange_b_b);

    if phi['model_log_sigma_sq'] is not None:      
      d.log_sigma_sq_phi_tensor[:] = -5;

    d.nu_theta[0] = 0.5;    
      
def init_model_arm1_rand1(theta,phi):
    flag_model_type = 'arm1';

    # Get the data from the model and put into d
    d = gd_vae.utils.DictAsMembers(get_data_of_model_select(theta,phi,flag_model_type));

    urange_w_a = urange_b_a = -1.0; urange_w_b = urange_b_b = 1.0;
    d.mu_theta_W1.uniform_(urange_w_a,urange_w_b);
    d.mu_theta_b1.uniform_(urange_b_a,urange_b_b);

    urange_w_a = urange_b_a = -1.0; urange_w_b = urange_b_b = 1.0;
    d.mu_theta_W2.uniform_(urange_w_a,urange_w_b);
    
    # Note: the [:] is important to get in-place value assignment (no history)    
    if theta['model_log_sigma_sq'] is not None:
      d.log_sigma_sq_theta_tensor[:] = -5;

    urange_w_a = urange_b_a = -1.0; urange_w_b = urange_b_b = 1.0;
    d.mu_phi_W1.uniform_(urange_w_a,urange_w_b);
    d.mu_phi_b1.uniform_(urange_b_a,urange_b_b);

    urange_w_a = urange_b_a = -1.0; urange_w_b = urange_b_b = 1.0;
    d.mu_phi_W2.uniform_(urange_w_a,urange_w_b);
    
    if phi['model_log_sigma_sq'] is not None:
      d.log_sigma_sq_phi_tensor[:] = -5;

    d.nu_theta[0] = 0.5;    
          
def test_model_errors(theta,phi,**extra_params):
    
    def avg_reconstruct_err(XX_batch,X_predict,**extra_params):
      """ Computes the relative errors."""
      norm_type, = tuple(map(extra_params.get,['norm_type']));

      if norm_type == 'L2':
        err = torch.mean(torch.sqrt(torch.sum(torch.pow(X_predict - XX_batch,2),1))).item(); # item() gets the scalar value
        ref = torch.mean(torch.sqrt(torch.sum(torch.pow(XX_batch,2),1))).item(); # item() gets the scalar value
        rel_err = err/ref;
      elif norm_type == 'L1':
        err = torch.mean(torch.sum(torch.abs(X_predict - XX_batch),1)).item(); # item() gets the scalar value
        ref = torch.mean(torch.sum(torch.abs(XX_batch),1)).item(); # item() gets the scalar value
        rel_err = err/ref;
      else:
        raise Exception("Not recognized, norm_type = " + norm_type);

      return rel_err, ref;

    # -- Compute Model Errors
    # Draw a batch of samples and determine the training error in reconstruction
    step_count,base_dir_training = tuple(map(extra_params.get,['step_count','base_dir_training']));
    sigma_list,device,flag_verbose = tuple(map(extra_params.get,['sigma_list','device','flag_verbose']));
    
    ref_dataset, = tuple(map(extra_params.get,['ref_dataset']));
                    
    flag_verbose = 1 if (flag_verbose is None) else flag_verbose;
            
    if sigma_list is None:
      sigma_list = [0.0]; # no noise case

    if flag_verbose >= 1:
      print("Compute estimate of the training accuracy in reconstructing configurations.");
              
    results = {};    
        
    # Compute the encoding + decoding to get prediction
    phi_model_mu = phi['model_mu']; theta_model_mu = theta['model_mu'];
    
    if flag_verbose >= 1:
      print("Direct reconstruction errors:")
    XX_batch_in = ref_dataset.samples_X_noise.detach(); XX_batch_in.requires_grad_(False);
    if flag_verbose >= 1:
      print("num_samples = " + str(XX_batch_in.shape[0]));
    Z_batch = phi_model_mu(XX_batch_in);
    X_predict = theta_model_mu(Z_batch);
    XX_batch_compare = ref_dataset.samples_X_target.detach();
    L1_err,L1_ref = avg_reconstruct_err(XX_batch_compare,X_predict,norm_type='L1');
    L2_err,L2_ref = avg_reconstruct_err(XX_batch_compare,X_predict,norm_type='L2');
    if flag_verbose >= 1:
      print("L2_err = %.4e"%L2_err);
      print("L1_err = %.4e"%L1_err);

    results.update({'L1_err_no_noise':L1_err,'L1_ref_no_noise':L1_ref,
                    'L2_err_no_noise':L2_err,'L2_ref_no_noise':L2_ref});
        
    s = {};
    s.update({'L2_err':L2_err,'L1_err':L1_err,'X_batch_in':XX_batch_in,
              'X_batch_compare':XX_batch_compare,'Z_batch':Z_batch,
              'X_predict':X_predict});
    study_I = 0;
    pickle_filename = '%s/test__training_error_no_noise_study_%.3d_%.7d.pickle'%(base_dir_training,study_I,step_count);
    if flag_verbose >= 1:
      print("pickle_filename = " + pickle_filename);
    fid = open(pickle_filename,'wb'); pickle.dump(s,fid); fid.close();   
    s = None; del s; # signal to clear memory                    
    
    count = 0;    
    L1_err_list = []; L2_err_list = [];
    for sigma in sigma_list:
        if flag_verbose >= 1:          
          print("Reconstruction errors with noise, X_in = X + noise:");   
          print("sigma = %.4e"%sigma);   
          
        XX_batch_in = ref_dataset.samples_X_noise.detach(); XX_batch_in.requires_grad_(False);
        XX_batch_in = XX_batch_in + sigma*torch.randn(XX_batch_in.size(),device=device);
        
        if flag_verbose >= 1:
          print("num_samples = " + str(XX_batch_in.shape[0]));
          
        Z_batch = phi_model_mu(XX_batch_in);
        X_predict = theta_model_mu(Z_batch);
        XX_batch_compare = ref_dataset.samples_X_target.detach(); XX_batch_compare.requires_grad_(False);

        L1_err,_ = avg_reconstruct_err(XX_batch_compare,X_predict,norm_type='L1');
        L1_err_list.append(L1_err);
        
        L2_err,_ = avg_reconstruct_err(XX_batch_compare,X_predict,norm_type='L2');
        L2_err_list.append(L2_err);

        if flag_verbose >= 1:
          print("L2_err = %.4e"%L2_err);
          print("L1_err = %.4e"%L1_err);

        s = {};
        s.update({'L2_err':L2_err,'L1_err':L1_err,'X_batch_in':XX_batch_in,
                  'X_batch_compare':XX_batch_compare,'Z_batch':Z_batch,
                  'X_predict':X_predict,'sigma':sigma});    
        pickle_filename = '%s/test__training_error_noise_study_%.3d_%.7d.pickle'%(base_dir_training,count,step_count);
        if flag_verbose >= 1:
          print("pickle_filename = " + pickle_filename);
        fid = open(pickle_filename,'wb'); pickle.dump(s,fid); fid.close();   
        s = None; del s; # signal to clear memory
        
        count += 1;
        
    results.update({'L1_err_list':np.array(L1_err_list),'L2_err_list':np.array(L2_err_list),'sigma_list':np.array(sigma_list)});    
    return results;

