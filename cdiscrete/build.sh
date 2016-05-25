ccache g++ -std=c++11 -g costs.cpp -c -larmadillo
ccache g++ -std=c++11 -g discrete.cpp -c -larmadillo
ccache g++ -std=c++11 -g driver.cpp -c -larmadillo
ccache g++ -std=c++11 -g function.cpp -c -larmadillo
ccache g++ -std=c++11 -g mcts.cpp -c -larmadillo
ccache g++ -std=c++11 -g misc.cpp -c -larmadillo
ccache g++ -std=c++11 -g policy.cpp -c -larmadillo
ccache g++ -std=c++11 -g transfer.cpp -c -larmadillo


ccache g++ -std=c++11 -g *.o -o driver -larmadillo
