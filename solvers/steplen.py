import numpy as np

def max_steplen(x,dir_x):
    neg = dir_x < 0
    if not np.any(neg):
        return 1.0
    step = np.min(-x[neg] / dir_x[neg])
    assert(step > 0)
    return step

def steplen_heuristic(x,dir_x,y,dir_y,scale):
    x_step = max_steplen(x,dir_x)
    y_step = max_steplen(y,dir_y)
    
    return min([1.0,scale*x_step, scale*y_step])

def sigma_heuristic(sigma,steplen):
    max_sigma = 0.999
    min_sigma = 0.1
    
    if(steplen >= 0.8):
        sigma *= 0.975  # Long step
    elif(steplen <= 0.2):
        sigma = 0.75 + 0.25*sigma
    elif (steplen < 1e-3):
        sigma = max_sigma
    return min(max_sigma,max(min_sigma,sigma))
