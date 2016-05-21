#include <iostream>
#include <armadillo>
#include <assert.h>
#include <set>

#include "misc.h"
#include "simulate.h"

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{

  SimulationOutcome res;
  simulate_test(res);
  
}
