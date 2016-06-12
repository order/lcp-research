#include <armadillo>
#include <assert.h>
#include <vector>

#include "discrete.h"

using namespace std;
using namespace arma;

int main(int argc, char** argv){

  RegGrid g;
  g.low = zeros<vec>(2);
  g.high = ones<vec>(2);
  g.num_cells = ones<uvec>(2);
 
  assert(verify(g));
  
  vector<vec> cuts;
  cuts.push_back(linspace<vec>(-0.1,1.1,50));
  cuts.push_back(linspace<vec>(-0.1,1.1,50));

  vec v = linspace<vec>(0,7,8);
  std::cout << v;
  
  mat P = make_points(cuts);

  vec F = interp_fn(v,P,g);

  mat I = reshape(F,50,50);
  I.save("test.img",raw_binary);
  
  
  return 0;
}
