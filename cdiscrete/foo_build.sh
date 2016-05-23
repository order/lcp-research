g++ -std=c++11 -O2 function.cpp -c -larmadillo
g++ -std=c++11 -O2 foo.cpp -c -larmadillo
g++ -std=c++11 -O2 discrete.cpp -c -larmadillo
g++ -std=c++11 -O2 misc.cpp -c -larmadillo

g++ -std=c++11 -O2 misc.o discrete.o function.o foo.o -o foo -larmadillo
