#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>

#include <armadillo>

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{
  mat M;
  M.load("test.h5",hdf5_binary);
  M = M.t();
  cout << size(M) << endl;
}
