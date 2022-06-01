from setuptools import setup, find_packages

setup(name='numpad_gym',
    version='0.1',
    description='Contains different numpad environments for reinforcement learning.',
    author='Simon Surland Andersen, Emil Møller Hansen',
    author_email='glq414@alumni.ku.dk,ckb257@alumni.ku.dk',
    packages=find_packages(where="src"),
    package_dir={"":"src"}
)
