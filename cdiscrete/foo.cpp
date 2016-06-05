#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>

#include <armadillo>

#include "discrete.h"

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{

  RegGrid g;
  g.low = zeros<vec>(2);
  g.high = ones<vec>(2);
  g.num_cells = ones<uvec>(2);

  
  mat v = mat("0,0,0,1,-10").t(); 
  mat P = mat("1,1");

  cout << interp_fns(v,P,g) << endl;;
  
}
