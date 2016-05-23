#include <iostream>
#include <armadillo>

#include <assert.h>
#include "misc.h"

using namespace std;
using namespace arma;

void foo(mat & A){
  cout << "In foo: " << A.memptr() << endl;
  A.fill(0);
}

int main(int argc, char** argv)
{
  vector<mat*> V;
  mat A = randu<mat>(5,3);
  V.push_back(&A);
  A.fill(0);
  cout << *V[0] << endl;
}
