#include <iostream>
#include <armadillo>

#include <assert.h>
#include "misc.h"
#include "function.h"

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{
  mat x = randu<mat>(10,2);
  ConstMultiFunction cmf = ConstMultiFunction(3,1.0);
  cout << cmf.f(x) << endl;

  ProbFunction pf = ProbFunction(&cmf);

   cout << pf.f(x) << endl;
 
}
