import subprocess
import sys,os

import numpy as np

import matplotlib.pyplot as plt

import stained_glass as sg
from utils.archiver import Unarchiver,read_medit_mesh
from kojima_solver import solve_lcp_file

from ctri_to_mesh_converter import convert_mesh_to_ctri,convert_ctri_to_mesh

GEN_COMMAND = './cdiscrete/dubins_gen'
REFINE_COMMAND = './cdiscrete/dubins_refine'
BASE_FILENAME = 'dubins'

##################################
# SYS CALLS ######################
##################################

def run_remesh(base_file,i):
    meshfile = base_file + '.' + str(i) + '.mesh'
    print "Running TETGEN on",meshfile

    assert os.path.isfile(meshfile) 

    cmd = './cdiscrete/tetgen -gq ' + meshfile
    print '#'*(len(cmd)+8)
    print 'RUNNING:', cmd
    print '-'*(len(cmd)+8)
    rc = subprocess.check_call([cmd],
                               shell=True,
                               stdout=sys.stdout,
                               stderr=sys.stderr)
    next_meshfile = base_file + '.' + str(i+1) + '.mesh'
    os.path.isfile(next_meshfile)
    return rc

def run_refine(base_file,i):
    in_name = base_file + '.' + str(i)
    out_name = base_file + '.' + str(i+1)
    assert os.path.isfile(in_name + '.ctri') 
    assert os.path.isfile(in_name + '.sol') 

    cmd = REFINE_COMMAND + ' --infile_base ' + in_name \
          + ' --outfile_base ' + out_name
    print '#'*(len(cmd)+8)
    print 'RUNNING:', cmd
    print '-'*(len(cmd)+8)
    return subprocess.check_call([cmd],
                                 shell=True,
                                 stdout=sys.stdout,
                                 stderr=sys.stderr)
def run_regen(base_file,i):
    filename = base_file + '.' + str(i)
    meshfile = filename + '.ctri'
    lcpfile = filename + '.lcp'
    assert os.path.isfile(meshfile) 

    cmd = GEN_COMMAND + ' --lcp ' + lcpfile\
          + ' --mesh ' + filename + '.ctri'
    print '#'*(len(cmd)+8)
    print 'RUNNING:', cmd
    print '-'*(len(cmd)+8)
    return subprocess.check_call([cmd],
                                 shell=True,
                                 stdout=sys.stdout,
                                 stderr=sys.stderr)

###############################
# PLOTTING ####################
###############################

def plot_solution(base_file, I):
    plt.figure(1)
    plt.clf()
    
    iterfile = base_file + '.' + str(I)
    meshfile = iterfile + '.mesh'
    solfile = iterfile + '.sol'
    assert os.path.isfile(meshfile) 
    assert os.path.isfile(solfile)

    print ""

    mesh = read_medit_mesh(meshfile)
    unarch = Unarchiver(solfile)
    f = unarch.p
    A = 3
    assert 0 == f.size % A
    N = f.size / A
    F = np.reshape(f,(N,A),'F')
    
    for a in xrange(A):
        plt.subplot(3,2,a+1,projection='3d')
        if a == 0:
            alpha_fn = lambda x: 0.5 * (1 - x)**1.5
            cmap = 'spectral_r'
            f = F[:-1,0]
        else:
            alpha_fn = lambda x: 0.1
            cmap = 'jet'
            f = np.argmax(F[:-1,1:],1)
            f = f.astype(np.double)
            f[f != (a-1)] = np.nan
        sg.plot_mesh(f,*mesh,
                     cmap=cmap,
                     alpha_fn = alpha_fn)
    plt.suptitle(base_file + str(I))

##########################################
# MAIN FUNCTION ##########################
##########################################
    
if __name__ == "__main__":
    (_,dir) = sys.argv
    base_file = dir + '/' + BASE_FILENAME
    
    # INITIAL LCP GENERATION
    # Consumes 0.ctri
    # Produces 0.lcp
    rc = run_regen(base_file,0)
    assert(0==rc)

    iterations = 5
    
    for I in xrange(0,2*iterations,2):
        print 'Iteration',I
        iterfile = base_file + '.' + str(I)
        refined_iterfile = base_file + '.' + str(I+1)
        tetgened_iterfile = base_file + '.' + str(I+2)

        # SOLVE THE LCP
        # Consumes I.lcp
        # Produces I.sol
        solve_lcp_file(iterfile + '.lcp')

        # PLOT THE SOLUTION
        print "Converting ",\
            iterfile + '.ctri to',\
            iterfile + '.mesh'
        convert_ctri_to_mesh(iterfile + '.ctri',
                             iterfile + '.mesh')
        
        # REFINE
        # Consumes I.sol, I.ctri
        # Produces I+1.ctri
        rc = run_refine(base_file,I)
        assert(0==rc)

        # Consumes I+1.ctri
        # Produces I+1.mesh
        print "Converting ",\
            refined_iterfile + '.ctri to',\
            refined_iterfile + '.mesh'        
        convert_ctri_to_mesh(refined_iterfile + '.ctri',
                             refined_iterfile + '.mesh')

        # Consumes I+1.mesh
        # Produces I+2.mesh
        rc = run_remesh(base_file,I+1)
        assert(0==rc)

        # Consumes I+2.mesh
        # Produces I+2.ctri (overwrites output of refine)
        convert_mesh_to_ctri(tetgened_iterfile + '.mesh',
                             tetgened_iterfile + '.ctri')
        print "Converting ",\
            tetgened_iterfile + '.mesh to',\
            tetgened_iterfile + '.ctri'   

        # Generate the LCP based on the refined mesh
        # Consumes I+2.ctri
        # Produces I+2.lcp
        rc = run_regen(base_file,I+2)
        assert(0==rc)
    plt.show()
