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
  arma_version ver;
  cout << "ARMA version: "<< ver.as_string() << endl;
  
  int D = 2;
  vec low = zeros<vec>(D);
  vec high = ones<vec>(D);
  uvec num = 10*ones<uvec>(D);

  UniformGrid grid = UniformGrid(low,high,num);

  Points P = {{0,0},{1,1},{1.1,0},{0,1.1},{-0.1,0},{0,-0.1},{0.5,0.5}};

  cout << "Points:\n" << P << endl;
  OutOfBounds oob = grid.points_to_out_of_bounds(P);
  cout << "OOB:\n" << oob;
  Coords coords = grid.points_to_cell_coords(P);
  cout << "Coords:\n" << coords;

  Indices cell_idx = grid.cell_coords_to_cell_indices(coords);
  cout << "Cell indices: " << cell_idx.t();

  Points low_points = grid.cell_coords_to_low_node(coords);
  cout << "Low node:\n" << low_points;
}
