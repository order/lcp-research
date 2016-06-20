CC=ccache g++
CFLAGS=-c -std=c++11 -O2 -g
IDIR= -I/usr/include/python2.7/
LIBS=-larmadillo -lpython2.7 -lboost_python -lboost_system
LDIR=

SOURCES= costs.cpp discrete.cpp function.cpp\
marshaller.cpp mcts.cpp misc.cpp policy.cpp reader.cpp simulate.cpp\
transfer.cpp value.cpp

OBJECTS=$(SOURCES:.cpp=.o)
EXEC=driver
TEST=discrete_tests

$(EXEC): $(OBJECTS) $(EXEC).o
	$(CC) $(OBJECTS) $(EXEC).o -o $@ $(LIBS) $(LDIR)

$(TEST): $(OBJECTS) $(TEST).o
	$(CC) $(OBJECTS) $(TEST).o -o $@ $(LIBS) $(LDIR)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@ $(IDIR)

clean:
	rm $(EXEC) $(TEST) *.o *.so *~ *.pyc callgrind.out.* test.mcts.*
