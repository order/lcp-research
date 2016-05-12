#include <iostream>
#include <armadillo>
#include <assert.h>
#include <set>
using namespace std;
using namespace arma;


int main(int argc, char** argv)
{
  mat M = zeros<mat>(5,3);
  cout << M + 1 << endl;
}
