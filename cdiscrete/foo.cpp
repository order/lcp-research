#include <iostream>
#include <armadillo>
#include <assert.h>
#include <set>
#include "misc.h"
using namespace std;
using namespace arma;

int main(int argc, char** argv)
{
  mat A = ones<mat>(2,2);
  rowvec b = rowvec("0,0");

  cout << dist(A,b) << endl;
}
