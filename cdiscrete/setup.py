#!/usr/bin/env python
 
from distutils.core import setup
from distutils.extension import Extension
 
setup(name="PackageName",
    ext_modules=[
        Extension("cDiscrete", ["binding.cpp","discrete.cpp"],
                  libraries = ["boost_python","armadillo"],
                  undef_macros = [ "NDEBUG" ],
                  extra_compile_args=['-std=c++11'])
    ])
