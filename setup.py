from distutils.core import setup
import io
import setuptools

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development
Operating System :: POSIX
Operating System :: Unix
"""

README = io.open("README.md", encoding="utf-8").read()

setup(
    name="chop-pytorch",
    description="Continuous and constrained optimization with PyTorch",
    long_description=README,
    long_description_content_type='text/markdown',
    version="0.0.2",
    author="Geoffrey Negiar",
    author_email="geoffrey_negiar@berkeley.edu",
    url="http://pypi.python.org/pypi/chop-pytorch",
    packages=["chop"],
    install_requires=["numpy", "scipy", "torch"],
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
)
