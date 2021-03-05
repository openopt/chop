from distutils.core import setup
import io
import setuptools

CLASSIFIERS = """\
Development Status :: 3 - Alpha
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
    name="chop",
    description="Library for continuous optimization using PyTorch",
    long_description=io.open("README.md", encoding="utf-8").read(),
    version="0.0.2",
    author="Geoffrey Negiar",
    author_email="geoffrey_negiar@berkeley.edu",
    url="http://pypi.python.org/pypi/chop-pytorch",
    packages=["chop"],
    install_requires=["numpy", "scipy", "torch", "easydict", "matplotlib", "tqdm"],
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    license="New BSD License",
)
