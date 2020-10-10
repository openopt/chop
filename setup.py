from distutils.core import setup
import io
import setuptools

CLASSIFIERS = """\
Development Status :: 2 - Pre-Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development
Operating System :: POSIX
Operating System :: Unix
"""

setup(
    name="constopt-pytorch",
    description="Library for constrained optimization using PyTorch",
    long_description=io.open("README.md", encoding="utf-8").read(),
    version="0.0.1",
    author="Geoffrey Negiar",
    author_email="geoffrey_negiar@berkeley.edu",
    url="http://pypi.python.org/pypi/constopt-pytorch",
    packages=["constopt"],
    install_requires=["numpy", "scipy", "torch"],
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    license="New BSD License",
)
