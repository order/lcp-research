#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>

#include <armadillo>
#include "grid.h"

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{
  int D = 5;
  vec low = zeros<vec>(D);
  vec high = ones<vec>(D);
  uvec num = 10*ones<uvec>(D);

  UniformGrid grid = UniformGrid(low,high,num);
 
}
