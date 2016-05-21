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
                                "discrete.cpp",
                                "misc.cpp",
                                "simulate.cpp"],
                  libraries = ["boost_python","armadillo"],
                  undef_macros = [ "NDEBUG" ],
                  extra_compile_args=['-std=c++11'])
    ])
