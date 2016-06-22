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
Marshaller marsh = Marshaller();
vec x = linspace<vec>(-10,12,256);
vec f = triangle_slope(x);
marsh.add_vec(x);
marsh.add_vec(f);
marsh.save("foo.dat");
}
