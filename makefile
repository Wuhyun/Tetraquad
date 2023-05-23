DIR := ${CURDIR}

PYTHON := python3

lib:
	${PYTHON} -m pip install -e tetraquad.py

clean:
	rm -rf build/*
	rm -rf lib/*
	rm -rf *.egg-info

