
from setuptools import setup, find_packages
  
setup(
    name='gpu_xray_scattering',
    version='0.1.0',
    author='Darren Hsu',
    packages=find_packages("gpu_xray_scattering"),
    url='https://github.com/darrenjhsu/gpu_xray_scattering',
    license='Apache 2',
    description='Debye X-ray scattering with GPU',
    long_description=open('README.md').read(),
)
