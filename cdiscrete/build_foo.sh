ccache g++ -std=c++11 -c foo.cpp
ccache g++ -std=c++11 -c io.cpp
#ccache g++ -std=c++11 -g -I/opt/local/include/ -c misc.cpp
#ccache g++ -std=c++11 -g -c grid.cpp
ccache g++ -std=c++11 foo.o io.o -lboost_system -larmadillo -o foo
