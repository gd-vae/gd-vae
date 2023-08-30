# Attempts to quick install the package using pip and sources on-line 

import subprocess;

# try a few different methods
flag_cont=True; nn = 1; 
while flag_cont:
  print("."*80);
  args="install torch torchvision torchaudio";
  cmd='pip ' + args;
  results1 = subprocess.run([cmd],shell=True);
  args="install -U gd-vae-pytorch";
  cmd='pip ' + args;
  results2 = subprocess.run([cmd],shell=True);
  if (results1.returncode == 0) and (results2.returncode == 0):
    flag_cont=False;flag_installed=True;
  I=I+1;
  if I>=nn:
    flag_cont=False;

# if installed peform a quick test (or report further installation steps needed) 
if flag_installed:
  print("-"*80);
  print("Test the package works:");
  print("-"*80);
  try:  
    print("."*80);
    import gd_vae_pytorch.tests.t1 as t1; t1.run(); 
    print("Looks like the package succeeded in installing.");
    print("To set up models and simulations, please see the examples");
    print("and documentation pages.");
  except Exception as err:
    print(err);
    print("Did not succeed in quick installing the package.");
    print("You may need to first install pytorch configured for your system.");
    print("Please see the documentation pages for more information.");
  print("."*80);
else:
  print("."*80);
  print("Did not succeed in quick installing the package.");
  print("You may need to first install pytorch configured for your system.");
  print("Please see the documentation pages for more information.");
  print("."*80);

