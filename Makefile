# Makefile
# for BPNN

main = main.cpp
bpnn.out: ${main} BPNN.cpp BPNN.h
	g++ -std=c++11 -O2 -Wall ${main} BPNN.cpp -o bpnn.out

.PHONY: clean

clean:
	rm bpnn.out
