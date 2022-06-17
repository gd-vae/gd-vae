"""
  .. image:: nn_py.png


  Methods for working with neural networks and other model classes for 
  use with backpropogation and training.
 
  If you find these codes or methods helpful for your project, 
  please cite our related work.

"""
# more information: http://atzberger.org/

import matplotlib; import matplotlib.pyplot as plt;
import numpy as np; import torch; import torch.nn;
import os,time,pickle;
from collections import OrderedDict; 

class MLP1(torch.nn.Module):
  r"""Creates a Multilayer Perceptron (MLP) with the 
  specified architecture. 
  """

  # constants
  ACT_TYPE_None = 'None'; ACT_TYPE_ReLU = 'ReLU'; 
  ACT_TYPE_RBF = 'RBF'; ACT_TYPE_Sigmoid = 'Sigmoid';
  
  def __init__(self,layer_sizes=None,layer_act_types=None,
               flag_bias = True,activation_func=None,
               device=None,flag_verbose=0):
    r"""Initializes Multilayer Perceptron (MLP) to have 
    a specified architecture. 

    Parameters:
      layer_size (list): size of each processing layer
      layer_act_types (list): activation function types to use after each layer
      flag_bias (boolean): if biases should be used for layers
      activation_func (function): default activation function to use
      device (torch.device): specified device on which to setup the model
      flag_verbose (int): level of messages to report during calculations

    """
    super(MLP1, self).__init__();
    
    self.flag_initialized = False;
    
    self.ACT_ReLU = torch.nn.ReLU(); self.ACT_Sigmoid = torch.nn.Sigmoid();

    if layer_sizes is not None:
      self.layer_sizes = layer_sizes; self.layer_act_types = layer_act_types; 
      self.flag_bias = flag_bias; self.depth = len(layer_sizes); 
    
      if activation_func is None: # set default 
        activation_func=self.ACT_ReLU;
    
      # create intermediate layers
      layer_dict = OrderedDict();
      NN = len(layer_sizes);
      for i in range(NN - 1):
        key_str = 'hidden_layer_%.4d'%(i + 1);
        layer_dict[key_str] = torch.nnLinear(layer_sizes[i], 
                                        layer_sizes[i+1],
                                        bias=flag_bias).to(device=device);
        if self.layer_act_types is not None:
          key_str = 'activation_func_%.4d'%(i + 1);
          if self.layer_act_types[i] == MLP1.ACT_TYPE_ReLU:
            layer_dict[key_str] = self.ACT_ReLU;
          elif self.layer_act_types[i] == MLP1.ACT_TYPE_RBF: 
            layer_dict[key_str] = self.ACT_RBF;
          elif self.layer_act_types[i] == MLP1.ACT_TYPE_Sigmoid:
            layer_dict[key_str] = self.ACT_Sigmoid;
          elif self.layer_act_types[i] == MLP1.ACT_TYPE_None:  
            pass; # no activation to add
          else:
            raise Exception("Not recognized layer_types[i] = " + str(layer_types[i]));
        else:
          if i < NN - 2: # last layer should be linear (or specified)
            key_str = 'activation_func_%.4d'%(i + 1);
            layer_dict[key_str] = activation_func;

      self.layers = torch.nnSequential(layer_dict); # uses ordered dictionary to create network
      self.flag_initialized = True;
    else:
      pass; # will be setup later
          
  def forward(self, input, params = None): 
    r"""Applies the Multilayer Perceptron (MLP) to the input data.
     
        Parameters:
          input (Tensor): the input for the MLP to process
          params (dict): parameters for the network (see examples and codes).

        Returns:
          **output** *(Tensor)* -- the evaluation of the network.  Returns tensor of size [batch,1].
    """
    # evaluate network with specified layers
    if params is None:
      eval = self.layers.forward(input);
    else:
      raise Exception("Not yet implemented for setting parameters.");
      
    return eval;

  def to(self, device,**extra_params):
    r"""Moves data to the specified device, gpu, cpu, or other."""
    super(MLP1, self).to(device,**extra_params);
    self.layers = self.layers.to(device,**extra_params);
    return self;
    
  def _get_save_data(self):
    r"""
    Internal method to setup model from saved data.
    """
    # Uses recommended PyTorch approach based on state_dict().
    s = {}; s.update({'layer_sizes':self.layer_sizes,'layer_act_types':self.layer_act_types,
                      'flag_bias':self.flag_bias,'depth':self.depth,
                      'state_dict':self.state_dict()});
    return s;

  def _set_from_save_data(self,s):
    r"""
    Internal method to setup model from saved data.
    """
    # load in the state dictionary
    self.layer_sizes = s['layer_sizes'];
    self.layer_act_types = s['layer_act_types'];
    self.flag_bias = s['flag_bias'];
    self.depth = s['depth'];

    self.load_state_dict(s['state_dict']);
    self.eval(); # for setup dropout or batch norm

    self.flag_initialized = True;

  def save_to_pickle(self,filename):
    r"""Save the weights to a pickle file."""
    # Uses recommended PyTorch approach based on state_dict().
    s = self._get_save_data();
    fid = open(filename,'wb'); pickle.dump(s,fid); fid.close();
                
  def load_from_pickle(self,filename):
    r"""Load the weights from pickle file.
        Assumes current architecture same as 
        the saved data.
    """
    fid = open(filename,'rb'); s = pickle.load(fid); fid.close();    
    self._set_from_save_data(s);

  @staticmethod
  def create_from_save_data(s):
    r"""
    Re-create the neural network from saved data.
    """
    mlp = MLP1(); mlp._set_from_save_data(s);
    return mlp;    
    
  @staticmethod
  def create_from_pickle(filename):
    r"""
    Re-create the neural network from saved data.
    """
    fid = open(filename,'rb'); s = pickle.load(fid); fid.close();
    mlp = MLP1.create_from_save_data(s);
    return mlp;
  
class AtzLearnableTensorFunc(torch.autograd.Function):
  '''
  Function for a learnable tensor (deprecated).  This was put in place to address issue with 
  an early version of pytorch.  This may be removed in future versions.
  '''

  @staticmethod
  def forward(ctx, X, tensor, params=None):
    r"""
    Function evaluation that returns a tensor.

    Paramaters:
      X (Tensor): input values
      tensor (Tensor): tensor to return independent of X.

    Returns:
      **output** *(Tensor)* -- a constant tensor with shape [num_samples,*tensor.shape].

    """

    info = {};    
    
    device = params['device'];

    num_samples = X.shape[0];
    v = torch.ones(num_samples,1,device=device);
    output = v*tensor; # assumes tensor.shape = [1,num_dim_z] -> output.shape = [num_samples,num_dim_z] using broadcasting

    info.update({'X':X,'tensor':tensor});
    ctx.atz_stored_for_later = info;

    return output;

  @staticmethod
  def backward(ctx, grad_output):
    r"""
    Computes the gradient.
    """
    info = ctx.atz_stored_for_later;
    X = info['X']; tensor = info['tensor'];
    grad_input = 1.0*grad_output;
    return 0*X, grad_input, None;

class AtzLearnableTensorLayer(torch.nn.Module):
  '''
  Module for storing a learnable tensor (deprecated).  This was put in place to address issue with 
  an early version of pytorch.  This may be removed in future versions.
  '''
  def __init__(self,**params):
    r"""
    Initialized the class.
    """
    super(AtzLearnableTensorLayer, self).__init__();
    
    self.params = params.copy();
    
    if 'device' in params:
      device = params['device'];
    else:
      self.params['device'] = torch.device('cpu');

    #self.tensor = None;
    if 'tensor_size' in params:      
      s = params['tensor_size'];
      self.params.pop('tensor_size', None); # delete key from dict      
      tensor = torch.zeros(*s,device=device);
      self.params['tensor'] = tensor;
    else:
      raise Exception("Must specify tensor_shape.");
    
  def forward(self,X):
    r"""
    Evaluate the tensor.
    """
    return AtzLearnableTensorFunc.apply(X,self.params['tensor'],self.params);

  def to(self,device):
    r"""
    Maps data to a specified device.
    """
    self.params['tensor'] = self.params['tensor'].to(device);
    self.params['device'] = device;
    return self;
    
  def extra_repr(self):
    r"""
    Give a string representation of the parameters.
    """
    # print information about this class    
    ss = 'AtzLearnableTensorLayer:\n';
    ss += 'tensor.shape = ' + str(self.params['tensor'].shape) + '\n';
    ss += 'tensor.device = ' + str(self.params['tensor'].device) + '';
    return ss;
