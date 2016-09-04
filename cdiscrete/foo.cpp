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
  Archiver arc;
  mat A = randu<mat>(5,5);

  arc.add_mat("A",A);
  arc.write("test.tar.gz");

  Unarchiver unarc = Unarchiver("test.tar.gz");
  mat B = unarc.load_mat("A");

  assert(approx_equal(A,B,"absdiff",1e-15));
}
