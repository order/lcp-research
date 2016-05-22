#include <iostream>
#include <armadillo>
#include <assert.h>
#include <set>

#include "misc.h"
//#include "simulate.h"

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{
  mat M = randu<mat>(7,3);
  cout << M << endl;
  cout << col_argmax(M) << endl;
}
