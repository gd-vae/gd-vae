import os,subprocess,json,copy,time;

#Define and create folder for saving
code_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_folder_path = os.path.join(code_folder, 'saved_data')
exp_folder_name = 'Final_Circle_Experiments'

num_runs_per_experiment = 1
list_of_experiments = [
    'VAE+Analytic_Projection',
    'VAE+Point_Cloud_Projection',
    'VAE+No_Projection', 
    'AE+Analytic_Projection', 
    'AE+No_Projection', 
    'VAE+2d',
    'VAE+10d'
    ]

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
    'num_epochs' : 400,
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
    run_params = copy.deepcopy(default_params)
    run_params['save_run_path'] = os.path.join(save_folder_path, exp_folder_name, run_name)
    run_params['data_folder_path'] = os.path.join(run_params['save_run_path'], 'data')
    if run_name_base == 'VAE+Analytic_Projection':
        pass
    elif run_name_base == 'VAE+Point_Cloud_Projection':
        run_params['use_analytic_projection_map'] = False
        run_params['use_point_cloud_projection_map'] = True
        run_params['num_points_in_cloud'] = 100
    elif run_name_base == 'VAE+No_Projection':
        run_params['use_analytic_projection_map'] = False
    elif run_name_base == 'AE+Analytic_Projection':
        run_params['mse_loss'] = True
    elif run_name_base == 'AE+No_Projection':
        run_params['use_analytic_projection_map'] = False
        run_params['mse_loss'] = True
    elif run_name_base == 'VAE+2d':
        run_params['use_analytic_projection_map'] = False
        run_params['latent_dim'] = 2
    elif run_name_base == 'VAE+10d':
        run_params['use_analytic_projection_map'] = False
        run_params['latent_dim'] = 10
    else:
        raise ValueError(f'Run Name {run_name_base} Not Recognized')
    return run_params
    
for run_name_base in list_of_experiments:
    for run_number in range(num_runs_per_experiment):
        run_name = run_name_base + f'_{run_number}'
        run_params = get_params_for_run(run_name_base, run_name, default_params)
        os.makedirs(run_params['save_run_path'])
        os.makedirs(run_params['data_folder_path'])

        param_filename = os.path.join(run_params['save_run_path'], "paramfile.json")
        with open(param_filename, 'w') as param_file:
            param_file.write(json.dumps(run_params))

        #Run Job
        output_filename = os.path.join(run_params['save_run_path'], 'output.out')
        train_script_filename = os.path.join(code_folder, 'train_model_periodic1.py')
        slurm_file_lines = [
            '#!/bin/bash -l',
            f'#SBATCH --job-name={run_name}',
            '#SBATCH --nodes=1',
            '#SBATCH --ntasks=1',
            '#SBATCH --time=3:00:00',
            f'#SBATCH --output={output_filename}',
            'conda activate VAE',
            f'python3 {train_script_filename} --paramFilename {param_filename}'
        ]
        with open('run_job.slurm', 'w') as slurm_file:
            slurm_file.write("\n".join(slurm_file_lines))
        subprocess.run(["sbatch", "run_job.slurm"])
        os.remove('run_job.slurm')
        time.sleep(5)


