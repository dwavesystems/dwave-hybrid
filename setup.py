import os
from io import open
from setuptools import setup, find_packages


# Load package info, without importing the package
basedir = os.path.dirname(os.path.abspath(__file__))
package_info_path = os.path.join(basedir, "hybrid", "package_info.py")
package_info = {}
try:
    with open(package_info_path, encoding='utf-8') as f:
        exec(f.read(), package_info)
except SyntaxError:
    execfile(package_info_path, package_info)


# Package requirements, minimal pinning
install_requires = ['six>=1.10', 'numpy>=1.15.0,<1.16.0',
                    'networkx', 'click>5', 'plucky>=0.4.3',
                    'dimod>=0.8.11', 'minorminer>=0.1.7', 'dwave-networkx>=0.6.6',
                    'dwave-system>=0.7.0', 'dwave-neal>=0.5.0', 'dwave-tabu>=0.2.0']

# Package extras requirements
extras_require = {
    'test': ['coverage', 'mock'],

    # python2 backports
    ':python_version == "2.7"': ['futures']
}

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]

setup(
    name=package_info['__packagename__'],
    version=package_info['__version__'],
    author=package_info['__author__'],
    author_email=package_info['__authoremail__'],
    description=package_info['__description__'],
    long_description=open('README.rst', encoding='utf-8').read(),
    url=package_info['__url__'],
    license=package_info['__license__'],
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=classifiers,
    zip_safe=False,
)
