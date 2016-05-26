ccache g++ -std=c++11 -g -c *.cpp -larmadillo -lhdf5 -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -I/usr/include/hdf5/serial/

ccache g++ -std=c++11 -g *.o -o foo -larmadillo -lhdf5 -L/usr/lib/x86_64-linux-gnu/hdf5/serial/ -I/usr/include/hdf5/serial/
