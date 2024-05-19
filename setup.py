from setuptools import Command, find_packages, setup

__lib_name__ = "TADGATE"
__lib_version__ = "1.0.0"
__description__ = "Uncovering topologically associating domains from three-dimensional genome maps with TADGATE"
__url__ = "https://github.com/zhanglabtools/TADGATE"
__author__ = "Dachang Dang"
__author_email__ = "dangdachang@163.com"
__license__ = "MIT"
__keywords__ = ["3D Genome", "topologically associating domains"]
__requires__ = ["requests",]

with open("README.rst", "r", encoding="utf-8") as f:
    __long_description__ = f.read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ['TADGATE'],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    long_description = __long_description__
)