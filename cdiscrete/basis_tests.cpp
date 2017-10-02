#include <assert.h>

#include <armadillo>
#include "basis.h"

using namespace std;
using namespace arma;

sp_mat build_shift_block(uint N, int shift, double p){
  assert(0 <= p);
  assert(p <= 1);
  assert(abs(shift) < N);

  vec shift_vec = (1 - p) * ones(N - abs(shift));
  vec wrap_vec = (1 - p) * one(abs(shift));
  
  return p * speye(N,N)
    + (1 - p) * spdiag(ones(N - abs(shift)), shift)
    + (1 - p) * spdiag(ones(abs(shift), abs(shift) - N));
}
