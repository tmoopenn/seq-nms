import setuptools
import numpy
from Cython.Build import cythonize

setuptools.setup(
    name             = 'seq-nms',
    version          = '0.0.1',
    description      = 'Implementation of seq-nms post-processing algorithm for video object detection.',
    url              = 'https://github.com/tmoopenn/seq-nms',
    author           = 'Troy Moo-Penn',
    packages         = setuptools.find_packages(),
    ext_modules      = cythonize("compute_overlap.pyx"),
    include_dirs     = [numpy.get_include()],
    setup_requires   = ["cython>=0.28", "numpy>=1.14.0"]
)