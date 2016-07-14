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

    alpha = -(x.dot(dir_y) + y.dot(dir_x)) / dir_x.dot(dir_y)
    print 'alpha:',alpha
    if alpha > 0:
        return min([1.0,alpha,scale*x_step, scale*y_step])
    
    return min([1.0,scale*x_step, scale*y_step])

def sigma_heuristic(sigma,steplen):
    if(1.0 >= steplen > 0.95):
        sigma *= 0.95  # Long step
    elif(0.1 >= steplen > 1e-3):
        sigma = 0.5 + 0.5*sigma
    elif (steplen <= 1e-3):
        sigma = 0.9 + 0.1*sigma
        #sigma = 0.9
    return min(0.999,max(0.1,sigma))
