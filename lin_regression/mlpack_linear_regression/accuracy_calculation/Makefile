all: main

main: main.o
	g++ -o main -lmlpack -larmadillo main.o #linking

main.o: main.cpp
	g++ -c -Wall -std=c++11 -pedantic main.cpp #compiling
