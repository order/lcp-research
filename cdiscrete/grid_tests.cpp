#include <armadillo>
#include <assert.h>

#include "grid.h"

using namespace std;

bool test_stride(){
  uvec stride = c_order_stride(uvec{3,2});
  assert(2 == stride.n_elem);
  assert(all(uvec{2,1} == stride));
}

int main(){
  cout << "Tests start..." << endl;
  test_stride();
}
