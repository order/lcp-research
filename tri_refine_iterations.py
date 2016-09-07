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
    out_name = base_file + '.' + str(i)
    cmd = CDISCRETE + '/refine --infile_base ' + in_name \
          + ' --outfile_base ' + out_name
    print '#'*(len(cmd)+8)
    print 'RUNNING:', cmd
    print '-'*(len(cmd)+8)
    subprocess.call([cmd],
                    shell=True,stdout=sys.stdout,stderr=sys.stderr)



if __name__ == "__main__":
    (_,dir) = sys.argv
    base_file = dir + '/di'
    run_di_gen_initial(base_file)

    mesh_viewer.plot_bare_mesh(base_file + '.0')
    plt.title('Initial mesh')
    plt.show()
    quit()
    for i in xrange(2):
        print 'Iteration',i
        solve_lcp_file(base_file + '.' + str(i) + '.lcp') # Generates .sol

        
        
        run_di_refine(base_file,i)
        break
