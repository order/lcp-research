#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>

#include <armadillo>

#include "marshaller.h"
#include "transfer.h"

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{
  arma_rng::set_seed_random();
  vec x = randn<vec>(5);

  cout << x.t();
  cout << min(x.t() > 0, x.t() < 1);
 
}
