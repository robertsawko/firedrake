from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
cmdclass = {}
cmdclass['build_ext'] = build_ext

import os
import sys


try:
    import petsc
    petsc_dir = (petsc.get_petsc_dir(), )
except ImportError:
    try:
        arch = os.environ.get("PETSC_ARCH")
        dir =  os.environ["PETSC_DIR"]
        if arch is not None:
            petsc_dir = (dir, os.path.join(dir, arch))
        else:
            petsc_dir = (dir, )
    except KeyError:
        sys.exit("""Couldn't find PETSc""")


include_dirs = [os.path.join(d, "include") for d in petsc_dir]
library_dirs = [os.path.join(petsc_dir[-1], "lib")]

try:
    import petsc4py
    include_dirs.append(petsc4py.get_include())
except ImportError:
    sys.exit("""Couldn't find petsc4py""")

try:
    import numpy
    include_dirs.append(numpy.get_include())
except ImportError:
    sys.exit("""Couldn't find numpy""")


extension = Extension("impl.patches", sources=["impl/patches.pyx"],
                      include_dirs=include_dirs,
                      libraries=["petsc"],
                      library_dirs=library_dirs,
                      runtime_library_dirs=library_dirs,
                      gdb_debug=True)


os.environ["CC"] = "mpicc"
os.environ["CXX"] = "mpic++"

setup(name="patches",
      cmdclass=cmdclass,
      ext_modules=[extension])
