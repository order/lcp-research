ccache g++ -std=c++11 -O2 costs.cpp -c -larmadillo
ccache g++ -std=c++11 -O2 discrete.cpp -c -larmadillo
ccache g++ -std=c++11 -O2 driver.cpp -c -larmadillo
ccache g++ -std=c++11 -O2 function.cpp -c -larmadillo
ccache g++ -std=c++11 -O2 mcts.cpp -c -larmadillo
ccache g++ -std=c++11 -O2 misc.cpp -c -larmadillo
ccache g++ -std=c++11 -O2 policy.cpp -c -larmadillo
ccache g++ -std=c++11 -O2 transfer.cpp -c -larmadillo


ccache g++ -std=c++11 -O2 *.o -o foo -larmadillo
