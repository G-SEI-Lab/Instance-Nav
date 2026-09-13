from setuptools import find_packages, setup


setup(
    name="instance-nav",
    version="0.1.0",
    description="Instance-Nav navigation and evaluation code",
    packages=find_packages(include=["instance_nav", "instance_nav.*"]),
    python_requires=">=3.9",
)
