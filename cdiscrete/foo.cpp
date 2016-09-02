#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>

#include <armadillo>
#include "io.h"

using namespace std;
using namespace arma;



int main(int argc, char** argv)
{
  
  sp_mat A = sp_mat(randu<mat>(5,5));
  cout << "Packing...";
  vec packed_A = pack_sp_mat<double>(A);
  cout << "." << endl;
  cout << "Unpacking...";
  sp_mat B = unpack_sp_mat<double>(packed_A);
  cout << "." << endl;

  assert(approx_equal(A,B,"absdiff",1e-15));
}
