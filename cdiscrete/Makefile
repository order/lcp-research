CC=ccache g++
CFLAGS=-c -std=c++11 -O2
IDIR=-I/usr/include/hdf5/serial/ -I/usr/include/python2.7/
LIBS=-larmadillo -lhdf5 -lpython2.7 -lboost_python -lboost_system
LDIR=-L/usr/lib/x86_64-linux-gnu/hdf5/serial

SOURCES= binding.cpp driver.cpp io.cpp policy.cpp value.cpp costs.cpp mcts.cpp simulate.cpp\
discrete.cpp function.cpp misc.cpp transfer.cpp

OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=driver

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) -o $@ $(LIBS) $(LDIR)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@ $(IDIR)
