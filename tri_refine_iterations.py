import subprocess
import sys

import numpy as np

import matplotlib.pyplot as plt

import tri_mesh_viewer as tmv
from utils.archiver import Unarchiver,read_shewchuk
from kojima_solver import solve_lcp_file

CDISCRETE = './cdiscrete'

def run_di_gen_initial(base_file):
    filename = base_file + '.0'
    cmd = CDISCRETE + '/di_gen --outfile_base ' + filename
    print '#'*(len(cmd)+8)
    print 'RUNNING:', cmd
    print '-'*(len(cmd)+8)
    return subprocess.check_call([cmd],
                                 shell=True,
                                 stdout=sys.stdout,
                                 stderr=sys.stderr)

def run_di_refine(base_file,i):
    in_name = base_file + '.' + str(i)
    out_name = base_file + '.' + str(i+1)
    cmd = CDISCRETE + '/di_refine --infile_base ' + in_name \
          + ' --outfile_base ' + out_name
    print '#'*(len(cmd)+8)
    print 'RUNNING:', cmd
    print '-'*(len(cmd)+8)
    return subprocess.check_call([cmd],
                                 shell=True,
                                 stdout=sys.stdout,
                                 stderr=sys.stderr)
def run_di_gen(base_file,i):
    filename = base_file + '.' + str(i)
    cmd = CDISCRETE + '/di_gen --outfile_base ' + filename\
          + ' --mesh_file ' + filename + '.tri'
    print '#'*(len(cmd)+8)
    print 'RUNNING:', cmd
    print '-'*(len(cmd)+8)
    return subprocess.check_call([cmd],
                                 shell=True,
                                 stdout=sys.stdout,
                                 stderr=sys.stderr)

def plot_solution(base_file, I):
    Modes = [('value',False),('policy',False),('agg',True)]
    plt.figure(2)
    plt.clf()
    for (i,(mode,log)) in enumerate(Modes):
        plt.subplot(2,2,i+1)
        tmv.plot_solution_mesh(base_file + '.' + str(I),
                               base_file + '.' + str(I) + '.sol',
                               mode,log)
        plt.title(mode)
    plt.suptitle(base_file + str(I))
    
def plot_refine_stats(base_file, I):
    plt.figure(3)
    plt.clf()
    (nodes,faces) = read_shewchuk(base_file + '.' + str(I))
    F = faces.shape[0]
    unarch = Unarchiver(base_file + '.' + str(I+1) + '.stats')

    number_of_vects = len(unarch.data)
    R = int(np.ceil(np.sqrt(number_of_vects)))
    C = int(np.ceil(number_of_vects / R))
    assert R*(C-1) < number_of_vects <= R*C
    for (i,(name,f)) in enumerate(unarch.data.items()):
        plt.subplot(R,C,i+1)
        assert((F,) == f.shape)
        tmv.plot_faces(nodes,faces,f)
        plt.title(name)
    plt.suptitle(base_file + '.' + str(I))

if __name__ == "__main__":
    (_,dir) = sys.argv
    base_file = dir + '/di'
    
    # INITIAL MESH AND LCP GENERATION
    rc = run_di_gen_initial(base_file)
    print rc
    assert(0==rc)

    plt.figure(1)
    plt.subplot(1,2,1)
    tmv.plot_bare_mesh(base_file + '.0')
    plt.title('Initial mesh')
    plt.draw()

    show_plots = True
    iterations = 4
    
    for I in xrange(iterations):
        print 'Iteration',I

        # SOLVE THE LCP
        solve_lcp_file(base_file + '.' + str(I) + '.lcp') # Generates .sol

        # PLOT THE SOLUTION
        if show_plots:
            plot_solution(base_file,I)
            plt.draw()

        # REFINE
        rc = run_di_refine(base_file,I)
        assert(0==rc)
        # PLOT THE HEURISTICS
        if show_plots:
            plot_refine_stats(base_file,I) 
            plt.draw()
        # Generate the LCP based on the refined mesh
        rc = run_di_gen(base_file,I+1)
        assert(0==rc)
    plt.figure(1)
    plt.subplot(1,2,2)
    tmv.plot_bare_mesh(base_file + '.' + str(iterations))
    plt.title('Final mesh')
    plt.show()
