r"""
  .. image:: geo_map_py.png

  Geometric maps for manifolds and related methods for computing gradients for training.
  Provides the *g-projection* maps discussed in the GD-VAE paper for training with
  manifold latent spaces within machine learning methods.

  If you find these codes or methods helpful for your project, please cite our related work.

"""
# more information: http://atzberger.org/

import torch,torch.nn,sklearn,sklearn.neighbors;

class PointCloudMapFunc(torch.autograd.Function):
    r"""
    Module layer which maps an input to nearest point in a manifold having a point cloud 
    representation.  This layer also handles computing the associated gradients 
    for use in backpropogation and training methods. 
    """
    @staticmethod
    def find_nearest_manifold_pt_kdtree(X0,params):
      r"""Finds the nearest point on the manifold using for efficiency a kdtree data structure 
            for the point cloud representation.

            Parameters:
              X0 (Tensor): input point to map to the manifold. Tensor of shape [num_pts,num_dim].
              
              params (dict): the parameters for the manifold map (matches find_k_nearest_neighs_kdtree()).

            Returns: 
              **x** *(Tensor)* -- giving the closest nearby point. Tensor of shape [num_pts,num_dim].    
      """
      x,I_x = PointCloudMapFunc.find_k_nearest_neighs_kdtree(X0,k=1,params=params);
      params['I_x'] = I_x;
      return x;
    
    @staticmethod
    def find_k_nearest_neighs_kdtree(X0,k,params):
        r"""Find the nearest neighbors on the manifold using for efficiency a kdtree data structure 
            for the point cloud representation.

            Parameters:
              X0 (Tensor): input point to map to the manifold. Tensor of shape [num_pts,num_dim].

              manifold_ptsX (Tensor): points of the manifold point cloud representation. 
                Tensor of shape [num_manifold_pts,num_dim].

              kdtree_params (dict): parameters for the kdtree methods.  Giving 'None' 
                will result in the use of default parameters. 

            Returns: 
              (tuple) containing
              
              **X_neighs** *(Tensor)* -- giving the closest nearby points.  Tensor of shape [num_pts,num_dim].

              **I_neighs** *(Tensor)* -- giving the indices of the closest points. Tensor of shape [num_pts,1].

        """
        KDTree = sklearn.neighbors.KDTree;
        
        r"""Finds nearest neighbors, assumes column vectors."""        
        flag_build_tree = False; # for now each time
        manifold_ptsX = params['manifold_ptsX'];

        if 'kdtree' in params:
          kdtree = params['kdtree'];
        else:
          kdtree = None;
        
        if kdtree is None:
          flag_build_tree = True;

        if flag_build_tree:
          if 'kdtree_params' in params:
            kdtree_params = params['kdtree_params'];
          else:
            kdtree_params = {'leaf_size':10,'metric':'euclidean'};                    
          leaf_size = kdtree_params['leaf_size'];
          metric = kdtree_params['metric'];
          #manifold_ptsX = params['manifold_ptsX'];
          kdtree = KDTree(manifold_ptsX.detach().cpu().numpy(),leaf_size=leaf_size,metric=metric);
            
          params['kdtree'] = kdtree;
          params['kdtree_params'] = kdtree_params;

        I_neighs = kdtree.query(X0.detach().cpu().numpy(),k=k,return_distance=False);
        I_neighs = I_neighs.squeeze(1);
        X_neighs = manifold_ptsX[I_neighs,:];

        return X_neighs, I_neighs;
    
    @staticmethod
    def forward(ctx, input, params=None):
        r"""
        Performs the projection mapping of the input points :math:`x` to the points :math:`z` on the manifold.
        
        Parameters:
          ctx (dict): pytorch context data structure.
          input (Tensor): points :math:`x` in the embedding space. Tensor of shape [num_samples,num_dim_x].
          params (dict): the parameters for the mapping

        Returns: 
          **output** *(Tensor)* -- points :math:`z` on the manifold obtained from mapping :math:`x`.  Tensor of size [num_samples,num_dim_x]. 

        **params** [members]

        =============================  =======================================
        **Property**                   **Description**
        -----------------------------  ---------------------------------------
        **u** (Tensor)                 coordinate parameterization for the 
                                       manifold points
        **coordinate_chart** (Tensor)  coordinate chart information
        **device** (torch.device)      for the hardware device as a specific 
                                       gpu, cpu, or other component
        **kdtree_params** (dict)       for the parameters for the kdtree 
                                       methods
        =============================  =======================================
          
        """
        info = {'params':params};
        #device = params['device'];
        device = input.device;

        X = input; # short-hand

        # -- solve the minimization problem (here we use a simple sample-point method)
        # use kd-tree to find closest point in the point-cloud representation of the manifold
        find_nearest_manifold_pt = params['find_nearest_manifold_pt']; # function to get closest point on the manifold
        find_nearest_manifold_pt_params = params['find_nearest_manifold_pt_params'];
        # for now we map to the nearest tabulated point, could also implement interpolations

        x = find_nearest_manifold_pt(X,find_nearest_manifold_pt_params);
        I_x = find_nearest_manifold_pt_params['I_x'];
        info.update({'x':x,'X':X,'I_x':I_x});
        output = x; # the collection of closest point outputs
        ctx.atz_stored_for_later = info;

        return output;

    @staticmethod
    def backward(ctx, grad_output):
        r"""
        Computes the gradients of the projection map.
        """                                
        info = ctx.atz_stored_for_later; params = info['params']; 
        x = info['x']; X = info['X']; I_x = info['I_x']; num_samples = x.shape[0]; num_dim_x = x.shape[1];
        device = x.device;
        
        # get surface information (could be coordinate charts specific to each x)
        get_manifold_sigma_info = params['get_manifold_sigma_info']; # function to get local sigm(u) and derivatives
        get_manifold_sigma_info_params = params['get_manifold_sigma_info_params']; # function to get local sigm(u) and derivatives
        get_manifold_sigma_info_params['I_x'] = I_x; # updated by recent closest point algorithm call 
        manifold_info = get_manifold_sigma_info(x,get_manifold_sigma_info_params);
        d_ui_sigma_k = manifold_info['d_ui_sigma_k']; d_ui_uj_sigma_k = manifold_info['d_ui_uj_sigma_k']; sigma_k = manifold_info['sigma_k']; 
        
        num_dim_u = d_ui_sigma_k.shape[1];

        # compute the tensors (note coordinate charts can be x-dependent)
        d_u_G = torch.zeros(num_samples,num_dim_u,num_dim_u,device=device);
        for i in range(0,num_dim_u):
          for j in range(0,num_dim_u):
            d_u_G[:,i,j] = torch.sum(d_ui_sigma_k[:,i,:]*d_ui_sigma_k[:,j,:],1) - torch.sum((X[:,:] - x[:,:])*d_ui_uj_sigma_k[:,i,j,:],1);

        # compute the tensors (note coordinate charts can be x-dependent)
        d_X_G = torch.zeros(num_samples,num_dim_u,num_dim_x,device=device);
        for i in range(0,num_dim_u):
            d_X_G[:,i,:] = -d_ui_sigma_k[:,i,:];

        # use Gauasian eliminiation to solve inverse [\nabla_u G]^{-1},
        # to obtain the final derivative
        b = d_X_G; A = d_u_G;
        #xx, LU = torch.solve(b,A); # dL/dX = -d_u_G^{-1}*d_X_G__d_x_L.  
        xx = torch.linalg.solve(A,b); # dL/dX = -d_u_G^{-1}*d_X_G__d_x_L.
        du_dX = -1*xx; # assumes shape = [num_samples,num_dim_u,num_dim_x]

        # -- for back-prop we have
        # dL/dX = du/dX*dx/du*dL/dx = -([nabla_u G]^{-1} \nabla_X G)*\nabla_u\sigma*dL/dx 

        # first calculate (dG/dX)*(dL/dx)
        dL_dx = grad_output; # assumes grad_output.shape = [num_samples,num_dim_x]
        dL_du = torch.zeros(num_samples,num_dim_u,device=device); # @optimize
        for i in range(0,num_dim_u):
          dL_du[:,i] = torch.sum(d_ui_sigma_k[:,i,:]*dL_dx[:,:],1);
                   
        dL_dX = torch.zeros(num_samples,num_dim_x,device=device); # @optimize
        for i in range(0,num_dim_x):
          dL_dX[:,i] = torch.sum(du_dX[:,:,i]*dL_du[:,:],1);
    
        grad_input = dL_dX;        
                        
        return grad_input, None;

class ManifoldPointCloudLayer(torch.nn.Module):
    r"""
    This layer maps an input onto a manifold having a point cloud 
    representation and handles computing the associated gradients for 
    use in backpropogation.
    """
    def __init__(self,params):
        r"""
        Parameters:
          params (dict): collection of parameters for the mapping.  

        **params** [members]

        =============================  =======================================
        **Property**                   **Description**
        -----------------------------  ---------------------------------------
        **u** (Tensor)                 coordinate parameterization for the 
                                       manifold points
        **coordinate_chart** (Tensor)  coordinate chart information
        **device** (torch.device)      for the hardware device as a specific 
                                       gpu, cpu, or other component
        **kdtree_params** (dict)       for the parameters for the kdtree 
                                       methods
        =============================  =======================================

        """
        super().__init__();
        self.params = params;

    def forward(self, input):
        r"""
        Performs the projection mapping of the input points :math:'x" to the points :math:'z' on the manifold.
        
        Parameters:
          input (Tensor): points :math:`x` in the embedding space. Tensor of shape [num_samples,num_dim_x].
          
        Returns:
          **output** *(Tensor)*: points :math:`z` on the manifold projected from :math:`x`.  Tensor of size [num_samples,num_dim_x]. 

        """
        # compute the periodic padding of the input
        return PointCloudMapFunc.apply(input,self.params);
    
    def to(self,device):
      r"""
      Maps the stored manifold points to the specified device.
      """
      self.params['manifold_ptsX'] = self.params['manifold_ptsX'].to(device);      
      return self;

    def extra_repr(self):
        r"""
        Gives a string representation for the parameters.
        """
        # print information about this class
        #return 'ManifoldPointCloudLayer: (no internal parameters)';
        return 'ManifoldPointCloudLayer: params.keys() = ' + str(self.params.keys());
    

class ManifoldDirectMapLayer(torch.nn.Module):
    r"""
    This layer projects an input onto a manifold having a direct 
    representation as an expression that can be backpropogated.    
    """
    def __init__(self,params):
        r"""
        Parameters:
          params (dict): the parameters of the map including

        **params** [members]

        ==========================  =======================================
        **Property**                **Description**
        --------------------------  ---------------------------------------
        **func_map** (function)     function for the direct mapping
        **params_map** (dict)       paramaters for the mapping function
        **device** (torch.device)   for the hardware device as a specific 
                                    gpu, cpu, or other component.  
        ==========================  =======================================

        """

        super().__init__();
        self.params = params;

    def forward(self, input):
        r"""
        Performs the projection mapping of the input points :math:`x` to the points :math:`z` on the manifold.
        
        Parameters:
          input (Tensor): points :math:`x` in the embedding space. Tensor of shape [num_samples,num_dim_x].
          
        Returns: 
          **output** *(Tensor)* -- points :math:`z` on the manifold projected from :math:`x`.  Tensor of size [num_samples,num_dim_x]. 

        """        
        func_map,func_map_params = tuple(map(self.params.get,['func_map','func_map_params']));
        
        # compute the direct mapping for the manifold
        output = func_map(input,func_map_params);

        return output;

    def to(self,device):
      r"""
      Currently nothing extra to do to map to a device.
      """
      return self;
    
    def extra_repr(self):
        r"""
        Gives a string representation for the parameters.
        """
        # print information about this class
        #return 'ManifoldPointCloudLayer: (no internal parameters)';
        return 'ManifoldDirectMapLayer: params = ' + str(self.params);

def map_clifford_torus(input,params):
    r"""
    Computes the clifford torus map as represented by a product-space of circles
    in :math:`R^{2n}`.  
    
    Parameters:
      input (Tensor): input points :math:`x` to map to the Clifford Torus.
      params_map (dict): parameters for the clifford torus map.  

    Returns: 
      **z** *(Tensor)* -- points mapped to the manifold

    **params_map** [members]

    ==========================  =======================================
    **Property**                **Description**
    --------------------------  ---------------------------------------
    **num_circles** (int)       (default is 2): for number of circles 
                                to use for the product-space 
    **device** (torch.device)   for the hardware device as a specific 
                                gpu, cpu, or other component.  
    ==========================  =======================================
      
    """
  
    # compute the mapping to Clifford torus
    num_circles,device = tuple(map(params.get,['num_circles','device']));
    
    if num_circles is None:
      num_circles = 2;
      
    if device is None:
      #device = torch.device('cpu');
      device = input.device;
    
    X = input; # short-hand

    num_dim_c = 2; num_dim_z = num_circles*num_dim_c; num_samples = X.shape[0];
    x = torch.zeros(num_samples,num_dim_z,device=device);
    for k in range(0,num_circles):
      v = X[:,k*num_dim_c:(k + 1)*num_dim_c];
      norm_v = torch.sqrt(torch.sum(torch.pow(v,2),1)).unsqueeze(1);
      x[:,k*num_dim_c:(k + 1)*num_dim_c] = v/norm_v;  # map to a unit circle

    return x; # the collection of closest point outputs

def map_sphere(input,params):
    r"""
    Computes the sphere map as represented in :math:`R^{n}`.  

    Parameters:
      input (Tensor): input points :math:`x` to map to the sphere.
      params_map (dict): parameters for the sphere map.  

    Returns: 
      **z** *(Tensor)* -- points mapped to the sphere.

    **params_map** [members]

    ==========================  =======================================
    **Property**                **Description**
    --------------------------  ---------------------------------------
    **sphere_r** (double)       (default is 1.0) radius of the sphere 
    **epsilon** (double)        (default is 1e-10) used to avoid 
                                dividing by zero
    **device** (torch.device)   for the hardware device as a specific 
                                gpu, cpu, or other component
    ==========================  =======================================

    """

    # compute the mapping to sphere
    sphere_r,epsilon,device = tuple(map(params.get,['sphere_r','epsilon','device']));

    if sphere_r is None:
      sphere_r = 1.0;

    if epsilon is None:
      epsilon = 1e-10;

    if device is None:
      #device = torch.device('cpu');
      device = input.device;

    X = input; # short-hand

    num_dim_z = X.shape[1]; # num_dim_x
    x = torch.zeros(num_samples,num_dim_z,device=device);
    norm_X = torch.sqrt(torch.sum(torch.pow(X,2),1)).unsqueeze(1);

    # we use epsilon to avoid division by zero
    x = X/(norm_X + epsilon);  # map to a sphere of radius r

    return x; # the collection of closest point outputs

