# Makefile
# for BPNN

main = test.cpp
bpnn: ${main} BPNN.cpp BPNN.h
	g++-5 -std=c++11 -Wall ${main} BPNN.cpp -o bpnn

.PHONY: clean

clean:
	rm bpnn
