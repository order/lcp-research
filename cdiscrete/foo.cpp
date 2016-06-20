#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>

#include <armadillo>


using namespace std;
using namespace arma;

int main(int argc, char** argv)
{

  mat M = randn<mat>(5,6);
  cout << M;
  cout << find(M < 0);
  M(find(M < 0)).fill(0);
  cout << M;
  
}
