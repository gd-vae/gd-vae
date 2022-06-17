r"""
  A collection of utility functions used in the package.   
 
  If you find these codes or methods helpful for your project, please cite our related work.

"""
# more information: http://atzberger.org/

import os,matplotlib,matplotlib.pyplot as plt,numpy as np,pickle,shutil;
from collections import OrderedDict;

def save_fig(base_filename,extra_label='',flag_verbose=True,
             dpi_set=200,flag_pdf=False,fig=None):
  r"""
  Saves figure to disk as image or pdf file.
  """    

  if fig is None:
    fig = plt.gcf();
    
  plt.figure(fig.number); # make current figure specified figure
    
  flagSimple = False;
  if flagSimple: # avoids flashing, just uses screen resolution (ignores dpi_set)
    if flag_pdf:    
      save_filename = '%s%s.pdf'%(base_filename,extra_label);
      if flag_verbose:
        print('save_filename = %s'%save_filename);
    
      plt.savefig(save_filename, format='pdf');

    save_filename = '%s%s.png'%(base_filename,extra_label);
    if flag_verbose:
      print('save_filename = %s'%save_filename);
    
    plt.savefig(save_filename, format='png');        
  else:
    fig = plt.gcf();  
    fig.patch.set_alpha(1.0);
    fig.patch.set_facecolor((1.0,1.0,1.0,1.0));
    
    if flag_pdf:    
      save_filename = '%s%s.pdf'%(base_filename,extra_label);
      if flag_verbose:
        print('save_filename = %s'%save_filename);
      
      plt.savefig(save_filename, format='pdf',dpi=dpi_set,facecolor=(1,1,1,1));
      
    save_filename = '%s%s.png'%(base_filename,extra_label);
    if flag_verbose:
      print('save_filename = %s'%save_filename);
    
    plt.savefig(save_filename, format='png',dpi=dpi_set,facecolor=(1,1,1,1));
    
class DictAsMembers(object):
  r"""
  Takes a dictionary with string keys and makes an object 
  that has these as variables (deprecated).  This allows for using 
  dot notation to de-reference dictionary items in the syntax.

  *Example:*

  >>> d = MakeDictMembers(d_dict);
  >>> d_dict['a'] = 1.0; # gets replaced with
  >>> d.a = 1.0;
  
  Should use this class sparingly and manage name spaces with care. 
  Using python 'map' is preferred method to dereference large number 
  of dictionary items. 

  """

  def __init__(self, d_dict):
    r"""
    Initilizes class to include dictionary items as members.

    Parameters:
      d_dict (dict): dictionary to convert to object members

    """

    self.__dict__.update(d_dict);    

def create_dir(dir_name):
  r"""
  Creates a directory on disk.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name);    
    
def rm_dir(dir_name):
  r"""
  Removes recursively a directory on disk.
  """
  if os.path.exists(dir_name):    
    shutil.rmtree(dir_name);
  else: 
    print("WARNING: rm_dir(): The directory to remove does not exist, dir_name = " + dir_name);


