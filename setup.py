from setuptools import find_packages, setup

setup(
    name="tetraquad",
    version="1.0.0",
    author="Wuhyun Sohn",
    author_email="wuhyun@kasi.re.kr",
    description="Tetraquad: numerical quadrature for tetrapyd integration",
    packages=find_packages(include=["tetraquad"]),
)
