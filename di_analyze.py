import numpy as np
import di.plotting as plotting

import matplotlib.pyplot as plt
import time
def load_data(filename):
    data = np.load(filename)
    primal = data['primal']
    dual = data['dual']
    return [primal,dual]

if __name__ == '__main__':
    [primal,dual] = load_data('data/di_traj.npz')
    
    Frames = plotting.split_into_frames_tensor(primal,3,527,21,25)    
    plotting.animate_frames(Frames[1:100,0,:,:])
