g++ -std=c++11 -O2 misc.cpp -c -larmadillo
#g++ -std=c++11 -O2 simulate.cpp -c -larmadillo
g++ -std=c++11 -O2 foo.cpp -c -larmadillo

g++ -std=c++11 -O2 simulate.o foo.o misc.o -o foo -larmadillo
