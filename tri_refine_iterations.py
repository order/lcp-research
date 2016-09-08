import subprocess
import sys

import matplotlib.pyplot as plt

import mesh_viewer
from kojima_solver import solve_lcp_file

CDISCRETE = './cdiscrete'

def run_di_gen_initial(base_file):
    filename = base_file + '.0'
    cmd = CDISCRETE + '/di_gen --outfile_base ' + filename
    print '#'*(len(cmd)+8)
    print 'RUNNING:', cmd
    print '-'*(len(cmd)+8)
    subprocess.call([cmd],
                    shell=True,stdout=sys.stdout,stderr=sys.stderr)

def run_di_refine(base_file,i):
    in_name = base_file + '.' + str(i)
    out_name = base_file + '.' + str(i+1)
    cmd = CDISCRETE + '/refine --infile_base ' + in_name \
          + ' --outfile_base ' + out_name
    print '#'*(len(cmd)+8)
    print 'RUNNING:', cmd
    print '-'*(len(cmd)+8)
    subprocess.call([cmd],
                    shell=True,stdout=sys.stdout,stderr=sys.stderr)

def run_di_gen(base_file,i):
    filename = base_file + '.' + str(i)
    cmd = CDISCRETE + '/di_gen --outfile_base ' + filename\
          + ' --mesh_file ' + filename + '.tri'
    print '#'*(len(cmd)+8)
    print 'RUNNING:', cmd
    print '-'*(len(cmd)+8)
    subprocess.call([cmd],
                    shell=True,stdout=sys.stdout,stderr=sys.stderr)



if __name__ == "__main__":
    (_,dir) = sys.argv
    base_file = dir + '/di'
    
    # INITIAL MESH AND LCP GENERATION
    run_di_gen_initial(base_file)
    plt.figure(1)
    plt.subplot(1,2,1)
    mesh_viewer.plot_bare_mesh(base_file + '.0')
    plt.title('Initial mesh')
    plt.draw()

    show_plots = True
    iterations = 8
    
    for I in xrange(iterations):
        iter_file = base_file + '.' + str(I)
        print 'Iteration',I

        # SOLVE THE LCP
        solve_lcp_file(iter_file + '.lcp') # Generates .sol
        Modes = [('value',False),('policy',False),('agg',True)]
        
        # PLOT THE SOLUTION
        if show_plots:
            plt.figure(2)
            plt.clf()
            for (i,(mode,log)) in enumerate(Modes):
                plt.subplot(2,2,i+1)
                mesh_viewer.plot_solution_mesh(iter_file,
                                               iter_file + '.sol',
                                               mode,log)
                plt.title(mode)
            plt.suptitle(iter_file)
            plt.draw()

        # REFINE
        run_di_refine(base_file,I)

        # PLOT THE HEURISTICS
        if show_plots:
            plt.figure(3)
            plt.clf()
            Names = ["val_diff.vec",
                     "flow_diff.vec",
                     "heuristic.vec",
                     "grad_x.vec",
                     "grad_y.vec",
                     "policy.uvec",
                     "flow_vol.vec"]
            next_iter_file = base_file + '.' + str(I+1)
            for (i,name) in enumerate(Names):
                plt.subplot(3,3,i+1)
                vec_file = next_iter_file + '.' + name
                mesh_viewer.plot_raw_binary_mesh(iter_file,
                                                 vec_file,
                                                 False)
                plt.title(name)
            plt.suptitle(iter_file)
            plt.draw()
            
        # Generate the LCP based on the refined mesh
        run_di_gen(base_file,I+1)

    plt.figure(1)
    plt.subplot(1,2,2)
    mesh_viewer.plot_bare_mesh(base_file + '.' + str(iterations))
    plt.title('Final mesh')
    plt.show()
