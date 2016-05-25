ccache g++ -std=c++11 -g -c foo.cpp -larmadillo -lhdf5 -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -I/usr/include/hdf5/serial/
ccache g++ -std=c++11 -g -c io.cpp -larmadillo -lhdf5 -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -I/usr/include/hdf5/serial/

ccache g++ -std=c++11 -g foo.o io.o -o foo -larmadillo -lhdf5 -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -I/usr/include/hdf5/serial/
