from setuptools import find_packages, setup

package_dir = {'': 'src'}

package_data = {'': ['*']}

setup_kwargs = {
    'name': 'natural_models',
    'version': '0.1.0',
    'description': 'Modeling and infering crop biomass dynamics.',
    'long_description': None,
    'author': 'Digital Gaia, Inc',
    'author_email': None,
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'package_dir': package_dir,
    'packages': find_packages(),
    'package_data': package_data,
    'python_requires': '>=3.8',
}

setup(**setup_kwargs)
