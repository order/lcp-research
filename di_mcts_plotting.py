import os
import numpy as np
import matplotlib.pyplot as plt

from utils import cdf_points

ROOT = os.path.expanduser('~/data/di/')
IMAGES = os.path.expanduser('~/data/images/')

def cdf_plot_file(ax,filename,*vargs,**kwargs):
    data = np.load(filename)
    (x,F) = cdf_points(data)
    ax.plot(x,F,*vargs,**kwargs)


def component_cdf_plot():
    fig = plt.figure()
    ax = plt.gca()
    cdf_plot_file(ax,ROOT + 'di.rollout.npy','-b',lw=2)
    cdf_plot_file(ax,ROOT + 'di.q_low.npy','-r',lw=2)
    cdf_plot_file(ax,ROOT + 'di.q_ref.npy','-g',lw=2)
    ax.set_xlabel('Discounted Cost')
    ax.set_title('Components of MCTS')
    ax.legend(['rollout','16x16','64x64'],loc='best')
    fig.savefig(IMAGES + 'mcts_components.png')
    plt.close()
    
def mcts32_cdf_plot():
    fig = plt.figure()
    ax = plt.gca()
    cdf_plot_file(ax,ROOT + 'di.rollout.npy','-b')
    cdf_plot_file(ax,ROOT + 'di.q_low.npy','-r')
    cdf_plot_file(ax,ROOT + 'di.q_ref.npy','-g')
    cdf_plot_file(ax,ROOT + 'di.mcts_low_32.npy','-k',lw=2)
    ax.set_xlabel('Discounted Cost')
    ax.set_title('MCTS 32 and components')
    ax.legend(['rollout','16x16','64x64', 'MCTS 32'],loc='best')
    fig.savefig(IMAGES + 'mcts_32.png')
    plt.close()
    
def mcts_cdf_plot():
    fig = plt.figure()
    ax = plt.gca()
    cdf_plot_file(ax,ROOT + 'di.rollout.npy','-k')
    labels = ['rollout']
    for x in [8,16,32,64,128,256]:
        cdf_plot_file(ax,ROOT + 'di.mcts_low_{0}.npy'.format(x),lw=2)
        labels.append('MCTS ' +str(x))
    ax.set_xlabel('Discounted Cost')
    ax.set_title('MCTS with various budgets')
    ax.legend(labels,loc='best')
    fig.savefig(IMAGES + 'mcts_all.png')
    plt.close()

def mcts_noq_cdf_plot():
    fig = plt.figure()
    ax = plt.gca()
    cdf_plot_file(ax,ROOT + 'di.rollout.npy','-k')
    labels = ['rollout']
    for x in [4,8,16,32,64,128,256]:
        cdf_plot_file(ax,ROOT + 'di.mcts_noq_{0}.npy'.format(x),lw=2)
        labels.append('MCTS ' +str(x))
    ax.set_xlabel('Discounted Cost')
    ax.set_title('MCTS without value information')
    ax.legend(labels,loc='best')
    fig.savefig(IMAGES + 'mcts_noq.png')
    plt.close()
    
def mcts_noflow_cdf_plot():
    fig = plt.figure()
    ax = plt.gca()
    cdf_plot_file(ax,ROOT + 'di.rollout.npy','-k')
    labels = ['rollout']
    for x in [4,8,16,32,64,128,256]:
        cdf_plot_file(ax,ROOT + 'di.mcts_noq_{0}.npy'.format(x),lw=2)
        labels.append('MCTS ' +str(x))
    ax.set_xlabel('Discounted Cost')
    ax.set_title('MCTS without flow information')
    ax.legend(labels,loc='best')
    fig.savefig(IMAGES + 'mcts_noflow.png')
    plt.close()
    
def mcts_neither_cdf_plot():
    fig = plt.figure()
    ax = plt.gca()
    cdf_plot_file(ax,ROOT + 'di.rollout.npy','-k')
    labels = ['rollout']
    for x in [4,8,16,32,64,128,256]:
        cdf_plot_file(ax,ROOT + 'di.mcts_noq_{0}.npy'.format(x),lw=2)
        labels.append('MCTS ' +str(x))
    ax.set_xlabel('Discounted Cost')
    ax.set_title('MCTS without value or flow information')
    ax.legend(labels,loc='best')
    fig.savefig(IMAGES + 'mcts_neither.png')
    plt.close()

component_cdf_plot()
mcts32_cdf_plot()  
mcts_cdf_plot()
mcts_noq_cdf_plot()
mcts_noflow_cdf_plot()
mcts_neither_cdf_plot()
