import os
from setuptools import setup, find_packages


# Load package info, without importing the package
basedir = os.path.dirname(os.path.abspath(__file__))
package_info_path = os.path.join(basedir, "hybrid", "package_info.py")
package_info = {}
with open(package_info_path, encoding='utf-8') as f:
    exec(f.read(), package_info)


# Package requirements, minimal pinning
install_requires = ['numpy>=1.19.1', 'networkx', 'click>5', 'plucky>=0.4.3',
                    'dimod>=0.10.13,<0.13',
                    # Include dwave-preprocessing, as dimod went back and forth with
                    # including it as a core dependency vs extra. Make it unbounded,
                    # as it's already constrained by dimod (if required).
                    'dwave-preprocessing',
                    'minorminer>=0.1.7', 'dwave-networkx>=0.8.8', 'dwave-system>=1.13.0',
                    'dwave-neal>=0.5.4', 'dwave-tabu>=0.2.0', 'dwave-greedy>=0.1.0']

# Package extras requirements
extras_require = {
    'test': ['coverage'],
}

python_requires = ">=3.7"

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
]

setup(
    name=package_info['__package_name__'],
    version=package_info['__version__'],
    author=package_info['__author__'],
    author_email=package_info['__author_email__'],
    description=package_info['__description__'],
    long_description=open('README.rst', encoding='utf-8').read(),
    url=package_info['__url__'],
    license=package_info['__license__'],
    packages=[pkg for pkg in find_packages() if pkg.startswith('hybrid')],
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=python_requires,
    classifiers=classifiers,
    zip_safe=False,
)
