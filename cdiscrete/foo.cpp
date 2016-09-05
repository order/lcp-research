#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>

#include <armadillo>
#include "misc.h"

using namespace std;
using namespace arma;


int main(int argc, char** argv)
{
  sp_mat A = sp_mat(diagmat(vec({1,2})));
  sp_mat C = sp_mat(diagmat(vec({3,4})));
  
  block_sp_row top {sp_mat(),A};
  block_sp_row bot {C,sp_mat()};
  block_sp_mat S {top,bot};

  bmat(S);
}
