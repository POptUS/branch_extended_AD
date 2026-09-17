from setuptools import find_packages, setup

setup(
    name='jaxnp_hash',
    version='0.1',
    packages=find_packages(),
    install_requires=[],
    tests_require=['pytest'],
    test_suite='tests',
    author='Jan Hückelheim',
    author_email='jhueckelheim@anl.gov',
    description='A package that wraps around JAX to implement autodiff for fixed control flow paths.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jhueckelheim/jaxnp_hash/',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
