print("="*80);
print("GD-VAE: Non-linear PDE Example");
print("");
print("http://atzberger.org");
print("-"*80);

# general packages
import sys,os,pickle,time,argparse,numpy as np;
import torch,torch.nn;

# GD-VAE packages
import gd_vae_pytorch as gd_vae,gd_vae_pytorch.vae,gd_vae_pytorch.geo_map;

# local packages
import pkg,pkg.model_utils as model_utils,pkg.datasets as datasets;

# de-reference for later convenience
PointCloudPeriodicProjWithTime = model_utils.PointCloudPeriodicProjWithTime;
analytic_periodic_proj_with_time = model_utils.analytic_periodic_proj_with_time;

PeriodicDataset = datasets.PeriodicDataset; 

Encoder_Fully_Connected = model_utils.Encoder_Fully_Connected;
Decoder_Fully_Connected = model_utils.Decoder_Fully_Connected;

dyvae_loss = gd_vae.vae.dyvae_loss;
mse_loss = torch.nn.MSELoss();

# load parameters for run
parser = argparse.ArgumentParser();
parser.add_argument('--param_filename','-p');
args = parser.parse_args();
print("");print("args.param_filename = " + str(args.param_filename));print("");
with open(args.param_filename, 'rb') as param_file:
  run_params = pickle.load(param_file);
  
print("run_params.keys() = " + str(run_params.keys()));
  
data_dir = run_params['data_folder_path'];

# passed into VAE loss function
extra_params = {};
extra_params['beta'] = run_params['beta']; # beta value in beta-VAE
extra_params['gamma'] = run_params['gamma']; # gamma for reconstruction term as regularization
extra_params['num_monte_carlo_samples'] = run_params['m1_mc']; 
extra_params['device'] = None;
extra_params['latent_prior_std_dev'] = torch.Tensor([run_params['latent_prior_std_dev']]);
extra_params['mse_loss'] = mse_loss;

# create training dataset
xi = torch.linspace(0,1.0,run_params['input_dim']+1)[0:-1];
train_data_params = {'time_step':run_params['time_step'],'noise':run_params['noise']};
train_dataset = PeriodicDataset(xi, run_params['train_num_samples'], train_data_params);
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=run_params['batch_size'],shuffle=True);

# encoder model
phi = {};
encoder = Encoder_Fully_Connected(run_params['encoder_size'], run_params['latent_dim']);
if (not run_params['use_analytic_projection_map']) and (not run_params['use_point_cloud_projection_map']):
  phi['model_mu'] = encoder.mean;
elif run_params['use_analytic_projection_map']:
  phi['model_mu'] = lambda input : analytic_periodic_proj_with_time(encoder.mean(input));
elif run_params['use_point_cloud_projection_map']:
  point_cloud_periodic_proj_with_time = PointCloudPeriodicProjWithTime(run_params['num_points_in_cloud']);
  phi['model_mu'] = lambda input : point_cloud_periodic_proj_with_time(encoder.mean(input));
phi['model_log_sigma_sq'] = encoder.log_variance;

# decoder model
decoder = Decoder_Fully_Connected(run_params['decoder_size'], run_params['latent_dim'])
theta = {'model_mu' : decoder.mean}

# latent map
latent_map = model_utils.latent_map_forward_in_time; # evolution map forward in time
latent_map_params = {'time_step':run_params['time_step']};

# training parameters
params_to_opt = []; # list of parameters to be optimized
params_to_opt += list(encoder.mean.parameters());
params_to_opt += list(encoder.log_variance.parameters());
params_to_opt += list(decoder.mean.parameters());
optimizer = torch.optim.Adam(params_to_opt, lr=run_params['learning_rate']);

# train
print("");
print("Training the models:");
num_steps = len(train_loader);
encoder.save_encoder_model(data_dir, epoch=0); decoder.save_decoder_model(data_dir, epoch=0);

print('.'*80);  
for epoch in range(run_params['num_epochs']):
  epoch_start_time = time.time();

  for i, (input,target) in enumerate(train_loader):

    # calculate loss funtion
    if not run_params['mse_loss']:
      loss = dyvae_loss(phi['model_mu'], phi['model_log_sigma_sq'], theta['model_mu'], 
                        latent_map, latent_map_params, input, target, **extra_params);
    elif run_params['mse_loss']:
      latent = phi['model_mu'](input);
      latent_ev = latent_map(latent);
      reconstructed = theta['model_mu'](latent);
      predicted = theta['model_mu'](latent_ev);
      loss = mse_loss(predicted, target) + run_params['gamma']*mse_loss(reconstructed, input);

    # perform gradient descent
    optimizer.zero_grad(); loss.backward(); optimizer.step();

    # report progress
    if ((i + 1) % 100) == 0 or i == 0:        
      msg = 'epoch: [%d/%d]; '%(epoch+1,run_params['num_epochs']);
      msg += 'batch_step = [%d/%d]; '%(i + 1,num_steps);
      msg += 'loss: %.3e; '%(loss.item());
      print(msg);

  # epoch finished    
  print("time taken: %.1e s"%(time.time()-epoch_start_time));
  print('.'*80);  

  if (epoch+1)%run_params['save_every_n_epoch'] == 0:
    encoder.save_encoder_model(data_dir, epoch+1);
    decoder.save_decoder_model(data_dir, epoch+1);

  #print("");

print("done training.");
print("="*80);

