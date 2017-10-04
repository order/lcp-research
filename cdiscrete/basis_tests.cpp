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
  vec wrap_vec = (1 - p) * ones(abs(shift));

  sp_mat shift_diag = spdiag(ones(N - abs(shift)), shift);
  assert(size(N,N) == size(shift_diag));
  sp_mat mod_diag = spdiag(ones(abs(shift)), abs(shift) - N);
  assert(size(N,N) == size(mod_diag));
  
  return p * speye(N,N) + (1 - p) * shift_diag + (1 - p) * mod_diag;
}

void test_build_shift_1(){
  cout << "test_build_shift_1...";
  cout.flush();
  uint N = 10;
  sp_mat shift = build_shift_block(N, 1, 0.1);

  assert(abs(N - accu(shift)) < 1e-9);
  for(uint i = 0; i < N; i++){
    assert(abs(1 - accu(shift.col(i))) < 1e-9);
  }
  cout << " PASSED." << endl;
}

void test_balance_basis_1(){
  cout << "test_balance_basis_1...";

  sp_mat shift = build_shift_block(3, 1, 0.1);
  vector<sp_mat> blocks;
  blocks.push_back(shift);
  
  vector<sp_mat> init;
  sp_mat val = sp_mat(3,1);
  val(0,0) = 1;
  init.push_back(val);
  init.push_back(sp_mat(3,0));
  
  vector<sp_mat> bal = balance_bases(init, blocks);
  assert(size(3,1) == size(bal.at(0)));
  assert(norm(val - bal.at(0)) < 1e-9);  // Same as input
  
  assert(size(3,1) == size(bal.at(1)));
  cout << " PASSED." << endl;

}

int main(){
  test_build_shift_1();
  
  test_balance_basis_1();
}
