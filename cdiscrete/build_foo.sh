ccache g++ -std=c++11 -g -c foo.cpp -larmadillo
ccache g++ -std=c++11 -g -c transer.cpp -larmadillo
ccache g++ -std=c++11 -g foo.o transfer.o -o foo -larmadillo
