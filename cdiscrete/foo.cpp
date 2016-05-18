#include <iostream>
#include <armadillo>
#include <assert.h>
#include <set>
using namespace std;
using namespace arma;

int main(int argc, char** argv)
{

  uint n = 3;
  uint d = 2;
  mat A = randu<mat>(n,d);
  
  uvec i = linspace<uvec>(0,n*n-1,n*n);
  i = i - (i / n) * n;
  cout << A << endl << endl;
  cout << i << endl << endl;

  mat w = A.rows(i);
  cout << w << endl;

}
