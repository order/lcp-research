import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from utils import cdf_points

ROOT = os.path.expanduser('~/data/hillcar/')
IMAGES = os.path.expanduser('~/Dropbox/Documents/Notes/images/hillcar/')

def cdf_plot_file(ax,filename,*vargs,**kwargs):
    data = np.load(filename)
    (x,F) = cdf_points(data)
    ax.plot(x,F,*vargs,**kwargs)

def component_cdf_plot():
    fig = plt.figure()
    ax = plt.gca()
    cdf_plot_file(ax,ROOT + 'hillcar.rollout.npy','-b',lw=2)
    cdf_plot_file(ax,ROOT + 'hillcar.q_low.npy','-r',lw=2)
    cdf_plot_file(ax,ROOT + 'hillcar.q_ref.npy','-g',lw=2)
    ax.set_xlabel('Discounted Cost')
    ax.set_title('Components of MCTS')
    ax.legend(['rollout','16x16','64x64'],loc='best')
    fig.savefig(IMAGES + 'mcts_components.png')
    plt.close()

def mcts_cdf_plot():
    fig = plt.figure()
    ax = plt.gca()
    cdf_plot_file(ax,ROOT + 'hillcar.rollout.npy','--k',lw=2.0)
    cdf_plot_file(ax,ROOT + 'hillcar.q_low.npy','--b',lw=2.0)
    cdf_plot_file(ax,ROOT + 'hillcar.q_ref.npy','--r',lw=2.0)

    labels = ['rollout','16x16','64x64']
    budgets = [4,8,16,32,64,128,256,512]
    B = len(budgets)
    colors = [cm.cool(x) for x in np.linspace(0,1,B)]
    
    for (budget,color) in zip(budgets,colors):
        cdf_plot_file(ax,ROOT + 'hillcar.mcts_low_{0}.npy'.format(budget),
                      lw=2.,color=color)
        labels.append('MCTS ' +str(budget))

    ax.set_xlabel('Discounted Cost')
    ax.set_title('MCTS with various budgets')
    ax.legend(labels,loc='best')
    fig.savefig(IMAGES + 'mcts_low.png')
    plt.close()

def mcts_pes_cdf_plot():
    fig = plt.figure()
    ax = plt.gca()
    cdf_plot_file(ax,ROOT + 'hillcar.rollout.npy','--k',lw=2.0)
    cdf_plot_file(ax,ROOT + 'hillcar.q_low.npy','--b',lw=2.0)
    cdf_plot_file(ax,ROOT + 'hillcar.q_ref.npy','--r',lw=2.0)

    labels = ['rollout','16x16','64x64']
    budgets = [4,8,16,32,64,128,256,512]
    B = len(budgets)
    colors = [cm.cool(x) for x in np.linspace(0,1,B)]
    
    for (budget,color) in zip(budgets,colors):
        cdf_plot_file(ax,ROOT + 'hillcar.mcts_noq_pes_{0}.npy'.format(budget),
                      lw=2.,color=color)
        labels.append('MCTS ' +str(budget))

    ax.set_xlabel('Discounted Cost')
    ax.set_title('Pessimistic MCTS with various budgets')
    ax.legend(labels,loc='best')
    fig.savefig(IMAGES + 'mcts_pes.png')
    plt.close()

def mcts_opt_cdf_plot():
    fig = plt.figure()
    ax = plt.gca()
    cdf_plot_file(ax,ROOT + 'hillcar.rollout.npy','--k',lw=2.0)
    cdf_plot_file(ax,ROOT + 'hillcar.q_low.npy','--b',lw=2.0)
    cdf_plot_file(ax,ROOT + 'hillcar.q_ref.npy','--r',lw=2.0)

    labels = ['rollout','16x16','64x64']
    budgets = [4,8,16,32,64,128,256,512]
    B = len(budgets)
    colors = [cm.cool(x) for x in np.linspace(0,1,B)]
    
    for (budget,color) in zip(budgets,colors):
        cdf_plot_file(ax,ROOT + 'hillcar.mcts_noq_opt_{0}.npy'.format(budget),
                      lw=2.,color=color)
        labels.append('MCTS ' +str(budget))

    ax.set_xlabel('Discounted Cost')
    ax.set_title('Optimistic MCTS with various budgets')
    ax.legend(labels,loc='best')
    fig.savefig(IMAGES + 'mcts_opt.png')
    plt.show()
    plt.close()

def mcts_noflow_cdf_plot():
    fig = plt.figure()
    ax = plt.gca()
    cdf_plot_file(ax,ROOT + 'hillcar.rollout.npy','--k',lw=2.0)
    cdf_plot_file(ax,ROOT + 'hillcar.q_low.npy','--b',lw=2.0)
    cdf_plot_file(ax,ROOT + 'hillcar.q_ref.npy','--r',lw=2.0)

    labels = ['rollout','16x16','64x64']
    budgets = [4,8,16,32,64,128,256,512]
    B = len(budgets)
    colors = [cm.cool(x) for x in np.linspace(0,1,B)]
    
    for (budget,color) in zip(budgets,colors):
        cdf_plot_file(ax,ROOT + 'hillcar.mcts_noflow_{0}.npy'.format(budget),
                      lw=2.,color=color)
        labels.append('MCTS ' +str(budget))

    ax.set_xlabel('Discounted Cost')
    ax.set_title('Noflow MCTS with various budgets')
    ax.legend(labels,loc='best')
    fig.savefig(IMAGES + 'mcts_noflow.png')
    plt.close()


def mcts_handicap_cdf_plot():
    fig = plt.figure()
    ax = plt.gca()
    cdf_plot_file(ax,ROOT + 'hillcar.mcts_low_128.npy','--k',lw=2.0)

    labels = ['MCTS 128']
    flavours = ['noflow',
              'noq_opt',
              'noq_pes']
    B = len(flavours)
    colors = [cm.cool(x) for x in np.linspace(0,1,B)]
    
    
    for (flavour,color) in zip(flavours,colors):
        cdf_plot_file(ax,ROOT + 'hillcar.mcts_{0}_128.npy'.format(flavour),
                      lw=2.0,color=color)
        labels.append('MCTS ' +str(flavour))

    ax.set_xlabel('Discounted Cost')
    ax.set_title('MCTS with various handicaps')
    ax.legend(labels,loc='best')
    fig.savefig(IMAGES + 'mcts_handicaps.png')
    plt.close()

def mcts_shocking_64_cdf_plot():
    fig = plt.figure()
    ax = plt.gca()
    cdf_plot_file(ax,ROOT + 'hillcar.mcts_low_512.npy','-k',lw=2.0)
    cdf_plot_file(ax,ROOT + 'hillcar.mcts_noq_pes_64.npy','-r',lw=2.0)
    cdf_plot_file(ax,ROOT + 'hillcar.mcts_noq_opt_64.npy','-g',lw=2.0)
    cdf_plot_file(ax,ROOT + 'hillcar.mcts_noflow_64.npy','-b',lw=2.0)

    ax.set_xlabel('Discounted Cost')
    ax.set_title('Unusual behavior in No Q MCTS 64')
    ax.legend(['MCTS 512',
               'No Q MCTS 64 Pesimistic',
               'No Q MCTS 64 Optimistic',
               'No flow MCTS 64'],loc='best')
    fig.savefig(IMAGES + 'mcts_shocking_64.png')
    plt.show()
    plt.close()

component_cdf_plot()
mcts_cdf_plot()
mcts_handicap_cdf_plot()
mcts_pes_cdf_plot()
mcts_opt_cdf_plot()
mcts_noflow_cdf_plot()
mcts_shocking_64_cdf_plot()
