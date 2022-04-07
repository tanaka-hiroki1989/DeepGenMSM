import numpy as np

def potential_function(x):
    return 4*(x**8+0.8*np.exp(-80*x*x)+0.2*np.exp(-80*(x-0.5)**2)+0.5*np.exp(-40*(x+0.5)**2))
