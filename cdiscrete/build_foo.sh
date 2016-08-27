ccache g++ -std=c++11 -g -c foo.cpp -larmadillo
ccache g++ -std=c++11 -g -c grid.cpp -larmadillo
ccache g++ -std=c++11 -g foo.o grid.o -o foo -larmadillo
