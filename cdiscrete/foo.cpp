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
  vec v = zeros<vec>(5);
  v(2)++;
  cout << v << endl;
}
