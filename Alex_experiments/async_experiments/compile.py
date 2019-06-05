from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("new_arch",  ["new_arch.py"])

#   ... all your modules that need be compiled ...

]

setup(
    name = 'ev_new_arch',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)
