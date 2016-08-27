ccache g++ -std=c++11 -g -c foo.cpp
ccache g++ -std=c++11 -g -c misc.cpp
ccache g++ -std=c++11 -g -c grid.cpp
ccache g++ -std=c++11 -g foo.o misc.o grid.o -larmadillo -o foo
