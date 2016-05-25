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
  mat data;
  RegGrid grid;

  import_data("../data/di/", data, grid);
  vec v = data.col(0);

  
  
}
