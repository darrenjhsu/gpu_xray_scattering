
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext 

class Build(build_ext):
    """Customized setuptools build command - builds protos on build."""
    def run(self):
        xs_command = ["make"]
        if subprocess.call(xs_command) != 0:
            sys.exit(-1)
        super().run() 

setup(
    name='gpu_xray_scattering',
    version='0.1.0',
    author='Darren Hsu',
    packages=["gpu_xray_scattering"],
#    has_ext_modules=lambda: True,
#    cmdclass={
#        'build_ext': Build,
#    },
    url='https://github.com/darrenjhsu/gpu_xray_scattering',
    license='Apache 2',
    description='Debye X-ray scattering with GPU',
    long_description=open('README.md').read(),
)
