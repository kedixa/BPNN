# Makefile
# for BPNN

bpnn: test.cpp BPNN.cpp BPNN.h
	g++-5 -std=c++11 -Wall test.cpp BPNN.cpp -o bpnn

.PHONY: clean

clean:
	rm bpnn
