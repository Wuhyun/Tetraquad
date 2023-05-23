DIR := ${CURDIR}

PYTHON := python3

lib:
	${PYTHON} -m pip install -e .

clean:
	rm -rf *.egg-info

