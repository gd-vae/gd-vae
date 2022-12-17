import os,sys,pickle,copy,time;
import gd_vae_pytorch as gd_vae,gd_vae_pytorch.utils;

#Define and save parameters to file
default_params = {
  'input_dim' : 100,
  'latent_dim' : 3,
  'time_step' : 0.25,
  'train_num_samples' : int(1e4),
  'test_num_samples' : int(1e4),
  'noise' : 0.02,
  'batch_size' : 100,
  'gamma' : 1, #biases pipeline to reconstruct functions
  'num_epochs' : int(4e2),
  'learning_rate' : 1e-4,
  'm1_mc' : 1, #for monte carlo estimate
  'beta' : 1e-4,
  'latent_prior_std_dev' : 1.0,
  'use_analytic_projection_map' : True,
  'use_point_cloud_projection_map' : False,
  'mse_loss' : False,
  'encoder_size': [100, 400, 400],
  'decoder_size': [400, 400, 100], 
  'save_every_n_epoch': 100,  
  'num_points_in_cloud': None 
}

def get_params_for_run(run_name_base, run_name, default_params):
  run_params = copy.deepcopy(default_params);
  run_params['save_run_path'] = os.path.join(save_folder_path, run_name);
  run_params['data_folder_path'] = os.path.join(run_params['save_run_path'],'data');
  if run_name_base == 'VAE__Analytic_Projection':
    pass;
  elif run_name_base == 'VAE__Point_Cloud_Projection':
    run_params['use_analytic_projection_map'] = False;
    run_params['use_point_cloud_projection_map'] = True;
    run_params['num_points_in_cloud'] = 100;
  elif run_name_base == 'VAE__No_Projection':
    run_params['use_analytic_projection_map'] = False;
  elif run_name_base == 'AE__Analytic_Projection':
    run_params['mse_loss'] = True;
  elif run_name_base == 'AE__No_Projection':
    run_params['use_analytic_projection_map'] = False;
    run_params['mse_loss'] = True;
  elif run_name_base == 'VAE__2d':
    run_params['use_analytic_projection_map'] = False;
    run_params['latent_dim'] = 2;
  elif run_name_base == 'VAE__10d':
    run_params['use_analytic_projection_map'] = False;
    run_params['latent_dim'] = 10;
  else:
    raise ValueError(f'Run Name {run_name_base} Not Recognized');
    
  return run_params;

# setup the parameter files
study_I = 1; study_name = 'study_%.4d'%study_I;
save_folder_path = './script_data/%s'%study_name;
gd_vae.utils.create_dir(save_folder_path);

num_runs_per_experiment = 1
list_of_experiments = [
  'VAE__Analytic_Projection',
  'VAE__Point_Cloud_Projection',
  'VAE__No_Projection', 
  'AE__Analytic_Projection', 
  'AE__No_Projection', 
  'VAE__2d',
  'VAE__10d'
  ];

for run_name_base in list_of_experiments:
  for run_number in range(num_runs_per_experiment):
    run_name = run_name_base + '_%.5d'%run_number;
    run_params = get_params_for_run(run_name_base,run_name,default_params);    
    
    gd_vae.utils.create_dir(run_params['save_run_path']);
    gd_vae.utils.create_dir(run_params['data_folder_path']);    
        
    param_filename = os.path.join(run_params['save_run_path'],"params.pickle");
    with open(param_filename, 'wb') as param_file:
      pickle.dump(run_params,param_file);
      
  
