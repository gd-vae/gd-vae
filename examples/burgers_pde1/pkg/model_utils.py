# general packages
import os,sys,numpy as np;
import torch,torch.fft; from torch import nn;
from collections import OrderedDict;

# GD-VAE packages
import gd_vae_pytorch as gd_vae,gd_vae_pytorch.vae,gd_vae_pytorch.geo_map;

# -- 
class ConvModel(nn.Module):
    """
    Series of convolutional layers followed by ReLU activations.
    """
    def __init__(self, conv_layers_info):
        """
        :param layers_info: Each element sequentially describes the convolutional layers, which is a list
            of [in_channels, out_channels, kernel_size, stride, padding]
        :type layers_info: list[list]
        """
        super().__init__();
        conv_layers = OrderedDict();
        for index, layer_info in enumerate(conv_layers_info):
            in_channels, out_channels, kernel_size, stride, padding = layer_info;
            conv_layers[f'convolution_{index:4d}'] = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, 
                                                                            padding=padding);
            if index != len(conv_layers_info)-1: #no activation last layer
                conv_layers[f'activation_function_{index:4d}'] = nn.ReLU();
        self._conv_model = nn.Sequential(conv_layers);

    def forward(self, input):
        """
        :param input: Input tensor with shape [batch dim, input dim]
        :return output_formatted: Output tensor with shape [batch dim, output dim]
        """
        input_formatted = input.unsqueeze(1); # add channel dim
        output = self._conv_model(input_formatted);
        output_formatted = output.squeeze(-1); # squeeze extra single data dim
        return output_formatted;


class ConvTransposeModel(nn.Module):
    """
    Series of transpose convolutional layers followed by ReLU activations.
    """
    def __init__(self, conv_layers_info):
        """
        :param layers_info: Each element sequentially describes the convolutional layers, which is a list 
            of [in_channels, out_channels, kernel_size, stride, padding]
        :type layers_info: list[list]
        :param output_dim: Input dimension of linear first layer
        """
        super().__init__();
        conv_layers = OrderedDict();
        for index, layer_info in enumerate(conv_layers_info):
            in_channels, out_channels, kernel_size, stride, padding = layer_info;
            conv_layers[f'convolutionT_{index:4d}'] = nn.ConvTranspose1d(in_channels, out_channels, 
                                                                         kernel_size, stride=stride, padding=padding);
            if index != len(conv_layers_info)-1: #no activation last layer
                conv_layers[f'activation_function_{index:4d}'] = nn.ReLU();
        self._conv_model = nn.Sequential(conv_layers);

    def forward(self, input):
        """
        :param input: Input tensor with shape [batch dim, input dim]
        :return output_formatted: Output tensor with shape [batch dim, output dim]
        """
        input_formatted = input.unsqueeze(-1); # add extra single data dim
        output = self._conv_model(input_formatted);
        output_formatted = output.squeeze(1); # squeeze channel dim
        return output_formatted;


class Encoder():
    """
    Creates encoder mean and log variance neural networks, which share convolutional layers and have a distinct linear last layer.
    """
    def __init__(self, conv_layers_info, output_dim):
        """
        :param layers_info: Each element sequentially describes the convolutional layers, which is a list
            of [in_channels, out_channels, kernel_size, stride, padding]
        :type layers_info: list[list]
        :param output_dim: Output dimension of linear last layer (same for mean and log variance)
        """
        self._conv_model = ConvModel(conv_layers_info);
        conv_model_output_dim = conv_layers_info[-1][1];
        self._mean_last_layer = nn.Linear(conv_model_output_dim, output_dim);
        self._log_variance_last_layer = nn.Linear(conv_model_output_dim, output_dim);

        # create neural networks
        self._mean = nn.Sequential(self._conv_model, self._mean_last_layer);
        self._log_variance = nn.Sequential(self._conv_model, self._log_variance_last_layer);

    def mean(self, input):
        """
        Applies encoder mean neural network to input

        :param input: Input tensor of shape [x_dim] or [batch_dim, x_dim]
        """
        input_formatted = self._format_input(input);
        output = self._mean(input_formatted);
        output_formatted = self._format_output(output, input.dim());
        return output_formatted;

    def log_variance(self, input):
        """
        Applies encoder log variance neural network to input

        :param input: Input tensor of shape [x_dim] or [batch_dim, x_dim]
        """
        input_formatted = self._format_input(input);
        output = self._log_variance(input_formatted);
        output_formatted = self._format_output(output, input.dim());
        return output_formatted;

    def _format_input(self, input):
        if input.dim() == 1:
            return input.unsqueeze(0); #add batch dim
        elif input.dim() == 2:
            return input;
    
    def _format_output(self, output, input_dim):
        if input_dim == 1:
            return output.squeeze(0); #squeeze batch dim
        elif input_dim == 2:
            return output;
    
    def save_encoder_model(self, save_location_path, epoch):
        """
        :param save_location_path: base path for saving and loading models
        """
        conv_layers_save_path = os.path.join(save_location_path, f'encoder_conv_layers_{epoch}.pickle');
        mean_save_path = os.path.join(save_location_path, f'encoder_mean_last_layer_{epoch}.pickle');
        log_variance_save_path = os.path.join(save_location_path, f'encoder_log_variance_last_layer_{epoch}.pickle');
        torch.save(self._conv_model.state_dict(), conv_layers_save_path);
        torch.save(self._mean_last_layer.state_dict(), mean_save_path);
        torch.save(self._log_variance_last_layer.state_dict(), log_variance_save_path);

    def load_encoder_model(self, save_location_path, epoch):
        """
        :param save_location_path: base path for saving and loading models
        """
        conv_layers_save_path = os.path.join(save_location_path, f'encoder_conv_layers_{epoch}.pickle');
        mean_save_path = os.path.join(save_location_path, f'encoder_mean_last_layer_{epoch}.pickle');
        log_variance_save_path = os.path.join(save_location_path, f'encoder_log_variance_last_layer_{epoch}.pickle');
        self._conv_model.load_state_dict(torch.load(conv_layers_save_path));
        self._mean_last_layer.load_state_dict(torch.load(mean_save_path));
        self._log_variance_last_layer.load_state_dict(torch.load(log_variance_save_path));


class Decoder():
    """
    Creates decoder mean neural network, which has single linear layer followed by convolutional layers.
    """
    def __init__(self, conv_layers_info, input_dim):
        """
        :param layers_info: Each element sequentially describes the convolutional layers, which is a list 
            of [in_channels, out_channels, kernel_size, stride, padding]
        :type layers_info: list[list]
        :param output_dim: Input dimension of linear first layer
        """
        self._conv_transpose_model = ConvTransposeModel(conv_layers_info);
        conv_model_input_dim = conv_layers_info[0][0];
        self._mean_first_layer = nn.Linear(input_dim , conv_model_input_dim);

        #Create neural network
        self._mean = nn.Sequential(self._mean_first_layer, self._conv_transpose_model)

    def mean(self, input):
        """
        Applies decoder mean neural network to input

        :param input: Input tensor of shape [latent_dim] or [batch_dim, latent_dim]
        """
        input_formatted = self._format_input(input);
        output = self._mean(input_formatted);
        output_formatted = self._format_output(output, input.dim());
        return output_formatted;
   
    def _format_input(self, input):
        if input.dim() == 1:
            return input.unsqueeze(0); # add batch dim
        elif input.dim() == 2:
            return input;

    def _format_output(self, output, input_dim):
        if input_dim == 1:
            return output.squeeze(0); # squeeze batch dim
        else:
            return output;
    
    def save_decoder_model(self, save_location_path, epoch):
        """
        :param save_location_path: base path for saving and loading models
        """
        conv_layers_save_path = os.path.join(save_location_path, f'decoder_conv_layers_{epoch}.pickle');
        mean_save_path = os.path.join(save_location_path, f'decoder_mean_first_layer_{epoch}.pickle');
        torch.save(self._conv_transpose_model.state_dict(), conv_layers_save_path);
        torch.save(self._mean_first_layer.state_dict(), mean_save_path);

    def load_decoder_model(self, save_location_path, epoch):
        """
        :param save_location_path: base path for saving and loading models
        """
        conv_layers_save_path = os.path.join(save_location_path, f'decoder_conv_layers_{epoch}.pickle');
        mean_save_path = os.path.join(save_location_path, f'decoder_mean_first_layer_{epoch}.pickle');
        self._conv_transpose_model.load_state_dict(torch.load(conv_layers_save_path));
        self._mean_first_layer.load_state_dict(torch.load(mean_save_path));


class Encoder_Fully_Connected():
    """
    Creates encoder mean and log variance neural networks.

    :param encoder_size: List of # of nodes in each hidden layer except output dimension
    """
    def __init__(self, encoder_size, output_dim):
        self.mean = FullyConnectedNetwork(encoder_size + [output_dim]);
        self.log_variance = FullyConnectedNetwork(encoder_size + [output_dim]);
    
    def save_encoder_model(self, save_location_path, epoch):
        """
        :param save_location_path: base path for saving and loading models
        """
        mean_save_path = os.path.join(save_location_path, f'encoder_mean_{epoch}.pickle');
        log_variance_save_path = os.path.join(save_location_path, f'encoder_log_variance_{epoch}.pickle');
        torch.save(self.mean.state_dict(), mean_save_path);
        torch.save(self.log_variance.state_dict(), log_variance_save_path);

    def load_encoder_model(self, save_location_path, epoch):
        """
        :param save_location_path: base path for saving and loading models
        """
        mean_save_path = os.path.join(save_location_path, f'encoder_mean_{epoch}.pickle');
        log_variance_save_path = os.path.join(save_location_path, f'encoder_log_variance_{epoch}.pickle');
        self.mean.load_state_dict(torch.load(mean_save_path));
        self.log_variance.load_state_dict(torch.load(log_variance_save_path));


class Decoder_Fully_Connected():
    """
    Creates decoder mean and log variance neural networks.
    """
    def __init__(self, decoder_size, input_dim):
        self.mean = FullyConnectedNetwork([input_dim] + decoder_size);
    
    def save_decoder_model(self, save_location_path, epoch):
        """
        :param save_location_path: base path for saving and loading models
        """
        mean_save_path = os.path.join(save_location_path, f'decoder_mean_{epoch}.pickle');
        torch.save(self.mean.state_dict(), mean_save_path);

    def load_decoder_model(self, save_location_path, epoch):
        """
        :param save_location_path: base path for saving and loading models
        """
        mean_save_path = os.path.join(save_location_path, f'decoder_mean_{epoch}.pickle');
        self.mean.load_state_dict(torch.load(mean_save_path));


class FullyConnectedNetwork(nn.Module):
    """
    Fully connected NN module.
    """
    def __init__(self, layer_sizes, flag_bias = True, activation_func=nn.ReLU()):
        super().__init__();
        layers_dict = OrderedDict();
        for layer_index in range(len(layer_sizes)-1):
            input_width = layer_sizes[layer_index];
            output_width = layer_sizes[layer_index+1];
            layers_dict['hidden_layer_%.4d'%(layer_index + 1)] = nn.Linear(input_width, output_width, bias=flag_bias);
            layers_dict['activation_func_%.4d'%(layer_index + 1)] = activation_func;
        layers_dict.pop('activation_func_%.4d'%(layer_index + 1)); # remove activation function for last layer
        self.layers = nn.Sequential(layers_dict);

    def forward(self, input):
        return self.layers.forward(input);
      

def latent_map_forward_in_time(input,params):
  r"""
  latent map forward_in_time
  """
  output = input.clone(); time_step = params['time_step'];
  output[...,-1] = output[...,-1] + time_step; # evolution map forward in time
  
  return output;


class BurgersPdeSolver():
  """
  Evolves functions according to Burgers' equation through Cole-Hopf transform. 
  Functions must be periodic, integrate to 0, and be defined on the x-axis unit interval [0,1].
  """
  def __init__(self, num_x_points, nu, time_step):
    self.nu = nu; # nu value in Burger's equation 
    self.time_step = time_step;
    self.nx = num_x_points;

  def evolve_burgers(self, u, time_step = None):
    """
    Evolves u forward in time according to Burgers equation $u_t = nu u_{xx} - uu_x$.

    :param u: Must be of shape [num_samples, nx]
    :param time_step: If a list / numpy array of length num_data, evolves each function by a different time step. 
        If a number, evolves all functions by the same time step.
    """
    if time_step is None:
      time_step = self.time_step;
    phi = self._cole_hopf(u);
    phi_ev = self._phi_evolver(phi, time_step = time_step);
    u_ev = self._cole_hopf_inverse(phi_ev);
    return u_ev;

  def _cole_hopf(self,u):
    """
    Computes cole-hopf transform on input u defined by $\phi(x) = CH( u(x) ) = 
    \exp[ (-1/(2*self.nu)) * \int_0^x u(x') dx']. Integration of u(x) occurs in fourier space. 
    """
    u = torch.complex(u, torch.zeros_like(u));
    u_fft = torch.fft.fft(u);
    u_int_fft = u_fft * self._get_integral_vec();
    u_int = torch.fft.ifft(u_int_fft);
    u_int = u_int.real;
    u_int = u_int - u_int[:,0,None]; # \int_0^x u(x') dx' = \int u(x')dx'|_{x'=x} - \int u(x')dx'|_{x'=0}
    phi = torch.exp( (-1 / (2*self.nu)) * u_int);
    return phi;

  def _phi_evolver(self, phi, time_step = None):
    """
    Evolves \phi forward in time according to the heat equation $u_t = self.nu u_{xx}$.
    """
    phi = torch.complex(phi, torch.zeros_like(phi));
    phi_fft = torch.fft.fft(phi);
    time_ev_vec = self._get_time_ev_vec(time_step, num_data = phi.shape[0]);
    phi_ev_fft = phi_fft * time_ev_vec;
    phi_ev = torch.fft.ifft(phi_ev_fft);
    phi_ev = phi_ev.real;
    return phi_ev;

  def _cole_hopf_inverse(self,phi):
    """
    Computes inverse cole-hopf transform on input \phi defined by $u(x) = CH^{-1}( \phi(x) ) = 
    -2 * self.nu * \frac{ \phi_x (x) }{ \phi (x) }. Differentiation of \phi(x) occurs in fourier space.
    """
    phi_complex = torch.complex(phi, torch.zeros_like(phi));
    phi_fft = torch.fft.fft(phi_complex);
    phi_x_fft = phi_fft * self._get_derivative_vec();
    phi_x = torch.fft.ifft(phi_x_fft);
    phi_x = phi_x.real;
    u = -2 * self.nu * torch.div(phi_x, phi);
    return u;

  def _get_derivative_vec(self):
    """
    Computes derivative_vec, which acts as a derivative operator in fourier space by taking complex product.
    """
    derivative_vec_real = torch.zeros(self.nx); derivative_vec_imag = torch.zeros(self.nx);
    for i in range(0,self.nx):
        if (i < self.nx/2):
            derivative_vec_imag[i] = 2.0*np.pi*i;
        else:
            derivative_vec_imag[i] = 2.0*np.pi*(i - self.nx);
    derivative_vec = torch.complex(derivative_vec_real, derivative_vec_imag);
    return derivative_vec;

  def _get_integral_vec(self):
    """
    Computes integral_vec, which acts as an integral operator in fourier space by taking complex product.
    """
    integral_vec_real = torch.zeros(self.nx); integral_vec_imag = torch.zeros(self.nx);
    for i in range(1,self.nx):
        if (i < self.nx/2):
            integral_vec_imag[i] = -1/(2.0*np.pi*i);
        else:
            integral_vec_imag[i] = -1/(2.0*np.pi*(i - self.nx));
    integral_vec = torch.complex(integral_vec_real, integral_vec_imag);
    return integral_vec;

  def _get_time_ev_vec(self, time_step, num_data = None):
    """
    Computes time_ev_vec, which evolves functions according to the heat equation $u_t = self.nu u_{xx}$ in fourier
    space by taking complex product.
    """
    if type(time_step) in [list, np.ndarray]:
      time_factor = torch.Tensor(time_step);
    else:
      time_factor = torch.full([num_data], time_step);
    time_ev_vec_real = torch.zeros((num_data, self.nx));
    time_ev_vec_imag = torch.zeros((num_data, self.nx));
    for i in range(0,self.nx):
      if (i<self.nx/2):
        time_ev_vec_real[:,i] = torch.exp(-4.0 * np.pi**2 * i**2 * self.nu * time_factor);
      else:
        time_ev_vec_real[:,i] = torch.exp(-4.0 * np.pi**2 * (i-self.nx)**2 * self.nu * time_factor);
    time_ev_vec = torch.complex(time_ev_vec_real, time_ev_vec_imag);
    return time_ev_vec;

# --
"""
Code for Analytic Periodic Projection
"""
def analytic_periodic_proj(input):
  """
  Projects 2d input onto unit periodic.
  """
  # adding epsilon to avoid zero division
  proj_input = torch.divide(input, 
                            torch.linalg.norm(input, dim=-1, 
                                              keepdim=True)+1e-5); 

  return proj_input;

def analytic_periodic_proj_with_time(input):
  """
  Projects first two dimensions onto unit periodic, while keeping last dimension (time) unchanged.
  """
  first_two_dim = input[...,0:2];
  time_dim = input[...,-1].unsqueeze(dim=-1);
  first_two_dim_proj = analytic_periodic_proj(first_two_dim);
  proj_input = torch.cat((first_two_dim_proj, time_dim), -1);
  
  return proj_input;

"""
Code for Point Cloud Periodic Projection
"""
class PointCloudPeriodicProj(nn.Module):
  """
  Projects 2d input onto manifold with a periodic topology by using a point 
  cloud representation.  This allows for use of general geometries.
  """
  def __init__(self, num_points_in_cloud,device=None):
    super().__init__();

    # setup manifold description as point cloud
    params_manifold = {};
    u = self._sample_periodic_coordinates(num_points_in_cloud);
    x = self._periodic_coordinate_chart(num_points_in_cloud);
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
		       
    params_map.update({'get_manifold_sigma_info':self._get_manifold_sigma_info});
    params_map.update({'get_manifold_sigma_info_params':{'manifold_chart_I':manifold_chart_I,
							 'manifold_chart_u':manifold_chart_u,
							 'manifold_ptsX':manifold_ptsX,
							 'params_manifold':params_manifold,
							 'device':device}});

    manifold_map = gd_vae.geo_map.ManifoldPointCloudLayer(params_map); # can be used as part of PyTorch pipeline with backprop
  
    self.point_cloud_periodic_proj = manifold_map; 

  def forward(self, input):
    return self.point_cloud_periodic_proj(input);

  def _sample_periodic_coordinates(self, num_sample_angles):
    """
    Uniformly samples angles on a periodic.
    """
    sample_angles = torch.zeros(num_sample_angles, 1);
    sample_angles[:,0] = torch.linspace(0, 2*np.pi, num_sample_angles);
    sample_angles = sample_angles.requires_grad_(True);
    
    return sample_angles;

  def _periodic_coordinate_chart(self, num_points_in_cloud):
    """
    Maps from angles on a periodic to points of a periodic embedding.
    """
    angles = torch.linspace(0,2*torch.pi,num_points_in_cloud); num_dim_periodic = 2;
    periodic_points = torch.zeros(angles.shape[0], num_dim_periodic);
    periodic_points[:,0] = torch.cos(angles); periodic_points[:,1] = torch.sin(angles);
    
    return periodic_points;

  def _get_manifold_sigma_info(self,x,params):
    # assumes x is 2D point on periodic
    # chooses local coordinate chart 
    #   chart 1: sigma(u) = [cos(u),sin(u)] 
    #
    # device = params['device'];  
    device = x.device;

    results = {}; num_samples = x.shape[0]; num_dim_x = x.shape[1]; num_dim_u = 1;

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
    # @optimize (can compute derivatives below analytically to make codes more efficient)
    with torch.enable_grad():
      sigma_k = torch.zeros(num_samples,num_dim_x,device=device);
      d_ui_sigma_k = torch.zeros(num_samples,num_dim_u,num_dim_x,device=device);
      d_ui_uj_sigma_k = torch.zeros(num_samples,num_dim_u,num_dim_u,num_dim_x,device=device);

      uu = u[II,:]; # get all u for the current chart
      uu = uu.detach().requires_grad_(True); # detach to make leaf variable of local comp. graph    
      sigma_k[II,0] = torch.cos(uu[:,0]); # sigma(u^*) = x (by assumption)    
      sigma_k[II,1] = torch.sin(uu[:,0]); 
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

class PointCloudPeriodicProjWithTime(nn.Module):
  r"""
  Projects the first two dimensions onto unit periodic through point cloud while 
  using the last dimension to represent time.
  """
  def __init__(self, num_points_in_cloud):
    super().__init__();
    self.point_cloud_periodic_proj = PointCloudPeriodicProj(num_points_in_cloud);

  def forward(self, input):
    first_two_dim = input[...,0:2];
    time_dim = input[...,-1].unsqueeze(dim=-1);
    first_two_dim_proj = self.point_cloud_periodic_proj(first_two_dim);
    proj_input = torch.cat((first_two_dim_proj, time_dim), -1);
    
    return proj_input;
  
  
