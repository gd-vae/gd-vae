import numpy as np; import matplotlib; import matplotlib.pyplot as plt; 
import os,sys,pickle;

# temporary path for testing
#sys.path.append('../../'); # insert in slot 1, so searched first
path_add = os.path.join('..','..');
#sys.path.insert(1,'../../'); # insert in slot 1, so searched first
sys.path.insert(1,path_add); # insert in slot 1, so searched first
path_add = '.';
#sys.path.insert(1,'../../'); # insert in slot 1, so searched first
sys.path.insert(1,path_add); # insert in slot 1, so searched first
#print("sys.path = " + str(sys.path));

import src; 
import src.nn; import src.utils; import src.vae; import src.geo_map; import src.log;
gd_vae = src;

def atz_get_lims_x(ll,figsize):
  a1 = ll[0]; b1 = ll[1]; c = figsize[1]/figsize[0]; a2 = c*a1; b2 = c*b1;
  xlim = [a1,b1];
  ylim = [a2,b2];
  return xlim,ylim;

def atz_get_lims_y(ll,figsize):
  a1 = ll[0]; b1 = ll[1]; c = figsize[0]/figsize[1]; a2 = c*a1; b2 = c*b1;  
  ylim = [a1,b1];
  xlim = [a2,b2];
  return xlim,ylim;

def plot_model_arm1_encoder_circle(theta,phi,X_batch,**extra_params):
        
  flag_legends = True;
  if 'flag_legends' in extra_params: flag_legends = extra_params['flag_legends'];        
    
  font = {'family' : 'DejaVu Sans',
          'weight' : 'normal',
          'size'   : 14}

  matplotlib.rc('font', **font);
                
  def plot_encoder_circle(Z_subset,sigma_phi_subset,nu,group_name=None):
      figsize = (8,6);
      fig1 = plt.figure(figsize=figsize,facecolor='white');

      ax = plt.gca();
      H_h_list = []; H_l_list =[];

      H, = plt.plot(Z_subset[:,0],Z_subset[:,1],'.',color=(0.1,0.1,0.9),markersize=8,alpha=0.5);
      H_h_list.append(H); H_l_list.append(group_name);
      
      H = plot_sigma(Z_subset,sigma_phi_subset,ax,color=(0.8,0.1,0.1),alpha=0.05);
      H_h_list.append(H['disk'][0]); H_l_list.append('std. dev. encoder');

      # prior distribution
      H = plot_sigma(0*Z_subset[0],nu,ax,color=(0.2,0.2,0.8),alpha=0.2);
      H_h_list.append(H['disk'][0]); H_l_list.append('std. dev. prior');

      ax.axis('equal');

      #plt.yscale('log');
      plt.xlabel(r'$z_1$');
      plt.ylabel(r'0');

      if 'ylim' in extra_params:
        ylim = extra_params['ylim'];
        xlim, ylim = atz_get_lims_y(ylim,figsize);
      else:
        xlim, ylim = atz_get_lims_x([-5.1,5.1],figsize);

      if 'xlim' in extra_params:
        xlim = extra_params['xlim'];
        xlim, ylim = atz_get_lims_x(xlim,figsize);
      else:    
        xlim, ylim = atz_get_lims_x([-5.1,5.1],figsize);

      plt.xlim(xlim); 
      plt.ylim(ylim);

      plt.title(r'Learned Encoding : $x \rightarrow z$');   

      if flag_legends:
        plt.legend(H_h_list, H_l_list,loc='upper right');
        
      plt.draw(); 

      return fig1;

  # plot the results
  Z_tensor = phi['model_mu'](X_batch);
  Z = Z_tensor.cpu().detach().numpy();

  sigma_sq_phi = torch.exp(phi['model_log_sigma_sq'](X_batch));
  sigma_phi = torch.sqrt(sigma_sq_phi).cpu().detach().numpy();
  nu = theta['nu'].cpu().detach().numpy();  
  nu = float(nu);
    
  fig1 = plot_encoder_circle(Z[:,0:2],sigma_phi[:,0:2],nu,'z group 1');
  fig2 = plot_encoder_circle(Z[:,2:4],sigma_phi[:,2:4],nu,'z group 2');

  fig_list = [fig1,fig2];

  return fig_list;

def plot_model_arm1_decoder_config_order(theta,phi,n1,n2,**extra_params):
        
  flag_legends = True;
  if 'flag_legends' in extra_params: flag_legends = extra_params['flag_legends'];        
    
  font = {'family' : 'DejaVu Sans',
          'weight' : 'normal',
          'size'   : 14}

  matplotlib.rc('font', **font);

  figsize = (12,9);
  fig,axs = plt.subplots(n1,n2,figsize=figsize,facecolor='white');
    
  fig.suptitle("Learned Decoder for Configurations", fontsize=14);

  # Loop over the encoder space (torus).
  pts_theta1 = torch.linspace(0,2*np.pi,n1,device=device);
  pts_theta2 = torch.linspace(0,2*np.pi,n2,device=device);  
  # Draw nearby configurations for the z values probed.
  for i1 in range(0,n1):
    for i2 in range(0,n2):
        ax = axs[i1][i2];
        ax.set_axis_off(); # hide the axis
        
        # compute the configuration for this choice of z1,z2
        theta1 = pts_theta1[i1]; theta2 = pts_theta2[i2];        
        Z_tensor = torch.tensor([torch.cos(theta1),torch.sin(theta1),
                                 torch.cos(theta2),torch.sin(theta2)],
                                device=device).unsqueeze(0);
        X_decode_tensor = theta['model_mu'](Z_tensor);
        X_decode = X_decode_tensor.cpu().detach().numpy();
        
        # plot the configuration for this encoding
        I = 0;
        X0 = np.array([0,0]);
        X1 = np.array([X_decode[I,0],X_decode[I,1]]);
        X2 = np.array([X_decode[I,2],X_decode[I,3]]);
        XX = np.vstack([X0,X1,X2]);

        H2, = ax.plot(XX[0,0],XX[0,1],'.',color=(0.1,0.1,0.1),ms=7,alpha=1.0);        
        H2, = ax.plot(XX[1:3,0],XX[1:3,1],'o',color=(0.8,0.1,0.8),
                      fillstyle=None,ms=5,alpha=0.5);        
        H2, = ax.plot(XX[0:2,0],XX[0:2,1],'-',color=(0.4,0.4,0.4),lw=1,alpha=0.5);      
        H2, = ax.plot(XX[1:3,0],XX[1:3,1],'-',color=(0.4,0.4,0.4),lw=1,alpha=0.5);

        ax.set_xlim([-2.1,2.1]);
        ax.set_ylim([-2.1,2.1]);
        
  plt.draw(); 
  
  return fig;

def plot_model_arm1_decoder(theta,phi,input,**extra_params):
    
  font = {'family' : 'DejaVu Sans',
          'weight' : 'normal',
          'size'   : 14}

  matplotlib.rc('font', **font);

  flag_legends = True;
  if 'flag_legends' in extra_params: flag_legends = extra_params['flag_legends'];    

  #X_batch = extra_paramsp['X_batch'];  
    
  figsize = (8,6);
  fig2 = plt.figure(figsize=figsize,facecolor='white');
  ax = plt.gca();
  H_h_list = []; H_l_list =[];

  X = input.cpu().numpy();
  X_batch = X; # reference to be the same as input (PJA recent) 
  
  Z_tensor = phi['model_mu'](input);
  Z = Z_tensor.cpu().detach().numpy();
    
  X_decode_tensor = theta['model_mu'](Z_tensor);
  X_decode = X_decode_tensor.cpu().detach().numpy();

  num_to_plot = int(1e0);
  for I in range(0,num_to_plot):
    X0 = np.array([0,0]);
    X1 = np.array([X_decode[I,0],X_decode[I,1]]);
    X2 = np.array([X_decode[I,2],X_decode[I,3]]);
    XX = np.vstack([X0,X1,X2]);
    
    H2, = plt.plot(XX[0,0],XX[0,1],'.',color=(0.1,0.1,0.1),ms=13,alpha=1.0);
    H_h_list.append(H2); H_l_list.append('X0');
        
    H2, = plt.plot(XX[1:3,0],XX[1:3,1],'o',color=(0.8,0.1,0.8),
                   fillstyle=None,ms=10,alpha=0.5);
    H_h_list.append(H2); H_l_list.append('X predict');
    
    H2, = plt.plot(XX[0:2,0],XX[0:2,1],'-',color=(0.4,0.4,0.4),lw=2,alpha=0.5);
    
    H2, = plt.plot(XX[1:3,0],XX[1:3,1],'-',color=(0.4,0.4,0.4),lw=2,alpha=0.5);
        
  num_to_plot = int(1e0);
  for I in range(0,num_to_plot):
    X0 = np.array([0,0]);
    X1 = np.array([X_batch[I,0],X_batch[I,1]]);
    X2 = np.array([X_batch[I,2],X_batch[I,3]]);
    XX = np.vstack([X0,X1,X2]);
            
    H2, = plt.plot(XX[1:3,0],XX[1:3,1],'.',color=(0.0,0.0,1.0),ms=13,alpha=0.9);
    H_h_list.append(H2); H_l_list.append('X target');

    H2, = plt.plot(XX[0:2,0],XX[0:2,1],'-',color=(0.4,0.4,0.4),lw=2,alpha=0.9);
    H_h_list.append(H2); H_l_list.append('rod 1');
    
    H2, = plt.plot(XX[1:3,0],XX[1:3,1],'-',color=(0.4,0.4,0.4),lw=2,alpha=0.9);
    H_h_list.append(H2); H_l_list.append('rod 2');
  
  ax.axis('equal');
  
  #plt.yscale('log');
  plt.xlabel(r'$x_1$');
  plt.ylabel(r'$x_2$');

  if 'ylim' in extra_params:
    ylim = extra_params['ylim'];
    xlim, ylim = atz_get_lims_y(ylim,figsize);
  else:
    xlim, ylim = atz_get_lims_x([-5.1,5.1],figsize);

  if 'xlim' in extra_params:
    xlim = extra_params['xlim'];
    xlim, ylim = atz_get_lims_x(xlim,figsize);
  else:    
    xlim, ylim = atz_get_lims_x([-5.1,5.1],figsize);
                    
  plt.xlim(xlim); 
  plt.ylim(ylim);
                    
  plt.title(r'Learned Decoding: $z \rightarrow x$');

  if flag_legends:
    plt.legend(H_h_list, H_l_list, loc='upper right');
  
  plt.draw();   
  
  return fig2;

def plot_model_arm1(theta,phi,input,**extra_params):
    
  fig_list = [];
  fig_name = [];
  
  if 'extra_params_encoder' in extra_params:
    extra_params_encoder = extra_params['params_encoder'];
  else:
    extra_params_encoder = {};
    
  if 'extra_params_decoder' in extra_params:
    extra_params_decoder = extra_params['params_decoder'];
  else:
    extra_params_decoder = {};

  flag_legends = True;
  if 'flag_legends' in extra_params: flag_legends = extra_params['flag_legends'];
        
  flag_manifold_map_encoder = False;      
  if 'flag_manifold_map_encoder' in extra_params:
    flag_manifold_map_encoder = extra_params['flag_manifold_map_encoder'];
            
  if flag_manifold_map_encoder:
    fig_name.append('encoder_circle_view1_group1');
    fig_name.append('encoder_circle_view1_group2');
    fig_list += plot_model_arm1_encoder_circle(theta,phi,input,xlim=[-3.5,3.5],
                                               flag_legends=flag_legends); # returns list of figs

  if flag_manifold_map_encoder:
    fig_name.append('decoder_config_order_view1');
    fig_list.append(plot_model_arm1_decoder_config_order(theta,phi,n1=10,n2=10,
                                                         flag_legends=flag_legends)); # returns list of figs

  fig_name.append('decoder_view1');
  fig_list.append(plot_model_arm1_decoder(theta,phi,input,xlim=[-3,3],
                                          flag_legends=flag_legends));

  plt.draw();   

  return fig_list,fig_name;

def plot_model(theta,phi,input,flag_model_type,**extra_params):
  if flag_model_type == 'arm1':
    fig_list,fig_names = plot_model_arm1(theta,phi,input,**extra_params);
  else:
    raise Exception("not recognized.");        
  return fig_list,fig_names;

def plot_loss(loss_step_list,loss_list,flag_tight_plot=False):
  fig = plt.figure(figsize=(8,6),facecolor=(1,1,1));

  plt.plot(loss_step_list,np.abs(loss_list),'b-');
  plt.yscale('log');
  plt.xlabel('step');
  plt.ylabel('|loss|');

  plt.title('Loss');
  if flag_tight_plot:
    plt.xlim([loss_step_list[0],loss_step_list[-1]]); # plot fills entire axis
  #plt.tight_layout();

  plt.draw();
  
  return fig;

def plot_info(d_info):
  d = gd_vae.utils.DictAsMembers(d_info);

  fig = plt.figure(figsize=(8,6),facecolor=(1,1,1));
  font = {'family' : 'DejaVu Sans',
              'weight' : 'normal',
              'size'   : 14}

  matplotlib.rc('font', **font);

  plt.axis('off');
  msg = 'step_count: %.6d'%d.step_count;
  plt.text(0.0,0.95,msg);

  msg = 'loss: %.3e'%d.loss;
  plt.text(0.0,0.88,msg);

  msg = 'epoch: %.3e'%d.epoch;
  plt.text(0.0,0.81,msg);
    
  plt.draw(); 
  
  return fig;
    
def plot_arm1_data(train_dataset,base_dir):
    font = {'family' : 'DejaVu Sans',
            'weight' : 'normal',
            'size'   : 14};

    matplotlib.rc('font', **font);
    
    XX = train_dataset.samples_X;  # reshape as needed
    num_dim = 2; num_samples = XX.shape[0]; num_pts = int(XX.shape[1]/num_dim);
    X = XX.reshape(num_samples,num_pts,num_dim);

    # Plot the configuration
    plt.figure(figsize=(8,6),facecolor='white');
    xx = X.detach().cpu().numpy();
    color_list = 0.2 + 0.7*np.random.rand(num_samples,3);
    flag_first = True;
    I_list = range(0,int(1e3));
    for I in I_list:
        val_color = color_list[I,:];
        if flag_first:
          label_str = 'X';
        else:
          label_str = None;
        plt.plot(xx[I,:,0],xx[I,:,1],'.',color=val_color,ms=8,alpha=0.7,label=label_str,zorder=2);

        xx = X.detach().cpu().numpy();
        num_dim = 2;
        num_pts = xx.shape[1]
        for k in range(0,num_pts-1):
          if k == 0 and flag_first:
            label_str = 'rod';
          else:
            label_str = None;
          plt.plot([xx[I,k,0],xx[I,k+1,0]],[xx[I,k,1],xx[I,k+1,1]],'-',lw=2,color=(0.4,0.4,0.4),alpha=0.3,label=label_str,zorder=1);

        flag_first = False;

    plt.xlabel(r'$x_1$'); plt.ylabel(r'$x_2$');
    plt.title('Configuration Samples');
    plt.axis('equal');
    
    ell_0 = 1.0;
    plt.xlim([-3.7*ell_0,3.7*ell_0]); plt.ylim([-3.7*ell_0,3.7*ell_0]);
    plt.legend();

    plt.draw();  
    
    base_filename = '%s/training_data'%(base_dir);
    gd_vae.utils.save_fig(base_filename,fig=plt.gcf(),flag_pdf=True);

def set_print_handle(print_handle):
  global print_log;
  print_log = print_handle;
  