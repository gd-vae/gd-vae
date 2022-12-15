from . import geometry as pkg_geometry;

import matplotlib.pyplot as plt;

def plot_klein_R3_immersion(params_klein,**kwargs):    
  x,u = pkg_geometry.sample_klein_bottle_points_R3(params_klein);

  from mpl_toolkits.mplot3d import Axes3D
  
  if 'ax_view' in kwargs:
    ax_view = kwargs['ax_view'];
  else:
    ax_view = [30,30];
    
  if 'skip_factor' in kwargs:
    skip_factor = kwargs['skip_factor'];
  else:
    skip_factor=17;
    
  if 'vec_x_lim' in kwargs:
    vec_x_lim = kwargs['vec_x_lim'];
  else:
    vec_x_lim=[(-4,4),(-4,4),(-4,4)];
    
  if 'alpha' in kwargs:
    alpha = kwargs['alpha'];
  else:
    alpha = 0.6;
    
  if 'ms' in kwargs:
    ms = kwargs['ms'];
  else:
    ms = 10;  
    
  # Plot the results (projection and the gradients)
  fig = plt.figure(figsize=(8,6),facecolor='white');
  ax = fig.add_subplot(111, projection='3d');

  xx = x.detach().cpu().numpy();
  
  plt.plot(xx[::skip_factor,1],xx[::skip_factor,2],xx[::skip_factor,0],'b.',label='manifold',alpha=alpha,ms=ms);

  ax.set_xlabel(r'$x_2$'); ax.set_ylabel(r'$x_3$'); ax.set_zlabel(r'$x_1$');   
  ax.set_xlim(vec_x_lim[0]); ax.set_ylim(vec_x_lim[1]); ax.set_zlim(vec_x_lim[2]);
  ax.view_init(elev=ax_view[0], azim=ax_view[1]);
  plt.title('Manifold');
  #plt.axis('equal');
  
  #plt.legend();

  plt.draw();

