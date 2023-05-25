# Tetraquad
A code for computing numerical quadrature rule for integrals over a tetrapyd domain. For mathematical details, please refer to [this paper](https://arxiv.org/abs/2305.14646).

## Dependencies

- Python3
- Python libraries:
    - numpy
    - scipy
    - mpmath
    - pandas
    - mayavi (optional for visualization)

## Installation
First of all, please check if you have installed the dependencies listed above. If your default Python3 command is not 'python3', change it in the makefile accordingly.

Clone the repository by, e.g.,
```
git clone https://github.com/Wuhyun/Tetraquad.git
```

Install the package by running
```
$ make
```
This installs the pacakge 'tetraquad' in development mode. To update the it, running
 ```
 git pull
 make
 ```
anytime will immediately update it.



## Quick Start

Please refer to the Jupyter notebook under examples/.
