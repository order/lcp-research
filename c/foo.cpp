#include <iostream>
#include <armadillo>
#include <assert.h>
#include <set>
using namespace std;
using namespace arma;

umat compare1(const mat & M, const rowvec & v){
  uint N = M.n_rows;
  return (M < repmat(v,N,1)); 
}

umat compare2(const mat & M, const rowvec & v){
  uint N = M.n_rows;
  uint D = M.n_cols;

  umat B = umat(N,D);
  uint elem;
  for(uint j = 0; j < D; j++){
    elem = v(j);
    for(uint i = 0; i < N; i++){
      B(i,j) = M(i,j) < elem;
    }
  }
  return B;
}

int main(int argc, char** argv)
{
  uint R = 2500;
  uint N = 10000;
  uint D = 10;
  
  mat M = randu<mat>(N,D);
  rowvec v = randu<rowvec>(D);

  for(uint k = 0; k < R; k++){
    compare2(M,v);
  }
  
}
