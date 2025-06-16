from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

ext_modules = cythonize(
    [
        Extension(
            name="galmex.cleaning",  # or "galmex.cleaning", make sure it matches your actual package
            sources=["galmex/cleaning.pyx"],
            include_dirs=[numpy.get_include()],
        )
    ],
    compiler_directives={
        'language_level': "3",
        'optimize.use_switch': True,
        'initializedcheck': False,
        'overflowcheck': False,
        'optimize.unpack_method_calls': True,
        'boundscheck': False,
        'profile': False,
        'infer_types': True,
        'cdivision_warnings': False,
        'cdivision': True,
        'wraparound': False,
    },
)

setup(
    name="galmex",
    version="1.0.5",
    description="Python package for measuring non-parametric morphological indices of galaxies",
    author="V. M. Sampaio",
    author_email="vitorms999@gmail.com",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vitorms99/galmex",
    packages=find_packages(include=["galmex", "galmex.*"]),
    setup_requires=["Cython", "numpy"],
install_requires=[
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "astropy",
    "scikit-image",
    "sep",
    "tqdm",
    "joblib",
    "filelock",
    ],
    ext_modules=ext_modules,
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
    'console_scripts': [
        'galmex = galmex.__main__:main',
    ],
    },
    python_requires=">=3.8",
    include_package_data=True,
    package_data={
        "galmex": [
            "examples/*",
            "Azure-ttk-theme/azure.tcl",
            "Azure-ttk-theme/theme/*",
            "manual.pdf",
            ],
    },    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

