#include <iostream>
#include <assert.h>
#include "discrete.h"

using namespace arma;

int main(int argc, char** argv)
{
  assert(argc == 3);
  uint T = atoi(argv[1]);
  uint N = atoi(argv[2]);
  uint D = 2;

  cube X = cube(D,N,T);
  X.slice(0).randn();

  mat physics = mat("1, 0.01; 0, 0.999");
  for(uint t = 1; t < T; t++){
    X.slice(t) = physics * X.slice(t-1);
  }
  //cout << X.slice(T-1) << endl;
}
