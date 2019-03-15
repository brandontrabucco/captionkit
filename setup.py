"""Setup script for captionkit."""
from setuptools import find_packages
from setuptools import setup
REQUIRED_PACKAGES = ['tensorflow-gpu', 'Pillow', 'numpy']
setup(name='captionkit', version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('captionkit')],
    description='Berkeley-CMU Image Captioning Toolkit')