#include <iostream>
#include <armadillo>
#include <assert.h>
#include <set>
using namespace std;
using namespace arma;

int main(int argc, char** argv)
{
  mat M = randu<mat>(10,2);
  rowvec l = rowvec("0.2 0.8");

  mat D = (M.each_row() < l);
  
  cout << D << endl;
  
}
