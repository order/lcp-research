CC=ccache g++
CFLAGS=-c -std=c++11 -O2 -g
IDIR=-I/usr/include/hdf5/serial/ -I/usr/include/python2.7/
LIBS=-larmadillo -lhdf5 -lpython2.7 -lboost_python -lboost_system
LDIR=-L/usr/lib/x86_64-linux-gnu/hdf5/serial

SOURCES= costs.cpp discrete.cpp driver.cpp function.cpp\
marshaller.cpp mcts.cpp misc.cpp policy.cpp simulate.cpp transfer.cpp\
value.cpp

OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=driver

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) -o $@ $(LIBS) $(LDIR)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@ $(IDIR)

clean:
	rm $(EXECUTABLE) *.o *.so *~ *.pyc callgrind.out.* test.mcts.*
