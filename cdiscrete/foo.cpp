#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>

#include <armadillo>

#include "marshaller.h"

using namespace std;
using namespace arma;

int main(int argc, char** argv)
{
  Marshaller marsh = Marshaller();
  mat A = ones<mat>(2,3);
  A(1,2) = 2;
  vec v = zeros<vec>(5);
  v(3) = 1;
  double d = 10;
  
  marsh.add_mat(A);
  marsh.add_vec(v);
  marsh.add_scalar(d);
  marsh.add_mat(A);
  marsh.save("test.bin");
  cout << marsh._header << endl;
  cout << marsh._data << endl;
  
  Demarshaller demarsh = Demarshaller("test.bin");
  
  mat B = demarsh.get_mat();
  vec u = demarsh.get_vec();
  double c = demarsh.get_scalar();
  mat C = demarsh.get_mat();
  
  cout << B << endl;
  cout << u << endl;
  cout << c << endl;
  cout << C << endl;
  
}
