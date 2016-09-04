#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>

#include <armadillo>
#include "io.h"

using namespace std;
using namespace arma;


typedef vector<vector<sp_mat>> block_sp_mat;
typedef vector<sp_mat> block_sp_row;
sp_mat bmat(const block_sp_mat & B){
  uint b_rows = B.size();
  assert(b_rows > 0);
  uint b_cols = B[0].size();
  assert(b_cols > 0);

  uvec rows = zeros<umat>(b_rows);  
  uvec cols = zeros<umat>(b_cols);
  
  for(uint i = 0; i < b_rows; i++){
    assert(b_cols == B[i].size());
    for(uint j = 0; j < b_cols; j++){
      if(rows[i]>0 and max(rows[i],B[i][j].n_rows)!= rows[i]){
	cerr << "[BMAT ERROR] Incompatible row dimensions" << endl;
	exit(1);
      }
      if(cols[j]>0 and max(cols[i],B[i][j].n_cols)!= cols[i]){
	cerr << "[BMAT ERROR] Incompatible col dimensions" << endl;
	exit(1);
      }
      rows[i] = max(rows[i],B[i][j].n_rows);
      cols[j] = max(cols[j],B[i][j].n_cols);      
    }
  }
  cout << rows.t();
  cout << cols.t();
  
  return sp_mat();
}

int main(int argc, char** argv)
{
  sp_mat A = speye(4,4);
  sp_mat B = speye(3,6);
  sp_mat C = speye(6,6);

  block_sp_row top {A,B};
  block_sp_row bot {sp_mat(),C};
  block_sp_mat S {top,bot};

  bmat(S);
}
