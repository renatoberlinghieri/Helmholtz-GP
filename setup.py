from setuptools import setup

setup(
    name="HelmholtzGP",
    version="1.0",
    description="Package for Modelling with Helmholtz GP",
    author="Renato Berlinghieri",
    author_email="renb@mit.edu",
    packages=["helmholtz_gp"],
    install_requires=["numpy", "torch", "matplotlib", "dataclasses", "scipy"],
)
