r"""
  Performs logging of messages for tracking runs.
"""
# more information: http://atzberger.org/

import sys,logging;

class PrintLog(object):
  r"""Redirects standard out and err output to 
  a file and also displays. 
  """

  def __init__(self,log_name=None,print_handler=None):
    r"""
    Initializes the logger.
    """
    self.logger = logging.getLogger(log_name);
    self.print_handler = print_handler;

  def write(self, msg,level=logging.INFO):
    r"""
    Writes data to the log and also echos in 
    a display if specified.
    """
    self.logger.info(msg);
    if self.print_handler is not None:
      self.print_handler(msg);
    #if self.sys_out is not None:
      #self.sys_out.write(msg);

  def flush(self):
    r"""
    Flushes the handler buffers.
    """
    for handler in self.logger.handlers:
        handler.flush();

def setup_log(base_dir):
  r"""
  Sets up logs.
  """
  # setup the logging  
  logging.basicConfig(format='%(message)s',
                      #level=logging.DEBUG, 
                      level=logging.INFO, 
                      filename='%s/main.log'%base_dir);

  # Redirect stdout and stderr
  #sys.stdout = PrintLog('stdout',sys.__stdout__.write);
  sys.stdout = PrintLog('stdout',None);
  logger = logging.getLogger('stdout');
  handler = logging.StreamHandler(stream=sys.__stdout__);
  handler.terminator = "";
  logger.addHandler(handler);
    
  #logger = logging.getLogger('stderr');
  #logger.addHandler(logging.StreamHandler(stream=sys.__stderr__))
  #sys.stderr = PrintLog('stderr',sys.__stderr__);

