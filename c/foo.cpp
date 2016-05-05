#include <iostream>
#include <armadillo>
#include <assert.h>
#include <set>
using namespace std;
using namespace arma;

uvec vec_mod( uvec a, uvec n){
  assert(a.n_elem == n.n_elem);
  return a - (a / n) % n; // '%' overloaded to be elem-mult
}

uvec vec_mod( uvec a, uint n){
  return a - (a / n) * n;
}

template <typename V>
V nans(uint n){
  V v = V(n);
  v.fill(datum::nan);
  return v;
}

int main(int argc, char** argv)
{
  mat A = randu<mat>(10,3);
  vec x = vec("0.2 0.2 0.2");
  set<uint> mask_set;

  cout << A << endl;
  uint D = x.n_elem;
  for(uint d = 0; d < D; ++d){
    uvec sr = find(A.col(d) < x[d]);
    mask_set.insert(sr.begin(),sr.end());
  }

  rowvec f = nans<rowvec>(D);
  for(set<uint>::const_iterator it = mask_set.begin();
      it != mask_set.end(); ++it){
    A.row(*it) = f;
  }
  cout << A << endl;
}
