# Makefile
# for BPNN

bpnn: main.cpp BPNN.cpp BPNN.h
	g++-5 -std=c++11 -Wall main.cpp BPNN.cpp -o bpnn

.PHONY: clean

clean:
	rm bpnn
