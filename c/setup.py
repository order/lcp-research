#!/usr/bin/env python
 
from distutils.core import setup
from distutils.extension import Extension
 
setup(name="PackageName",
    ext_modules=[
        Extension("discrete", ["discrete.cpp"],
                  libraries = ["boost_python","armadillo"],
                  extra_compile_args=['-std=c++11'])
    ])
