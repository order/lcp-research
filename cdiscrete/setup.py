#!/usr/bin/env python
 
from distutils.core import setup
from distutils.extension import Extension
import os
import sys


ON_LINUX = "linux" in sys.platform

if ON_LINUX:
    os.environ['CC'] = 'ccache gcc'
    
setup(name="PackageName",
    ext_modules=[
        Extension("cDiscrete", ["binding.cpp",
                                "costs.cpp",
                                "discrete.cpp",
                                "function.cpp",
                                "io.cpp",
                                "mcts.cpp",
                                "misc.cpp",
                                "policy.cpp",
                                "simulate.cpp",
                                "transfer.cpp",
                                "value.cpp"],
                  include_dirs=["/usr/include/hdf5/serial/"],
                  library_dirs=["/usr/lib/x86_64-linux-gnu/hdf5/serial"],
                  libraries = ["boost_python","armadillo","hdf5"],
                  undef_macros = [ "NDEBUG" ],
                  extra_compile_args=['-std=c++11'])
    ])
