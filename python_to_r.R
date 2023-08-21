library(reticulate)
use_python('') #El código r debesaber dónde esta instalado python
py_run_string('def Psq (x):
              value= x+x
              return(value)')
py$Psq(3)