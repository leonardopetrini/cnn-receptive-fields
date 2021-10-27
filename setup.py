"""
Installation script
"""
from setuptools import setup, find_packages

setup(
    name='receptivefield',
    packages=find_packages(exclude=["build"])
)