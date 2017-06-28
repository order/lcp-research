#include <armadillo>
#include <assert.h>

#include "grid.h"

using namespace std;

/*
 * Helper function tests
 */

bool test_stride_2d(){
  uvec stride = c_order_stride(uvec{3,2});
  assert(2 == stride.n_elem);
  assert(all(uvec{2,1} == stride));
  cout << "Finished test_stride_2d" << endl;
}
bool test_stride_3d(){
  uvec stride = c_order_stride(uvec{2,3,4});
  assert(3 == stride.n_elem);
  assert(all(uvec{12,4,1} == stride));
  cout << "Finished test_stride_3d" << endl;
}
bool test_shift_2d(){
  uvec shift = c_order_cell_shift(uvec{3,3});
  assert(4 == shift.n_elem);
  assert(all(uvec{0,3,1,4} == shift));
  cout << "Finished test_shift_2d" << endl;
}
bool test_shift_3d(){
  uvec grid = uvec{2,3,4};
  uvec shift = c_order_cell_shift(grid);
  
  assert(8 == shift.n_elem);
  assert(0 == shift(0));
  assert(sum(c_order_stride(grid)) == shift(7));
  cout << "Finished test_shift_3d" << endl;
}

/*
 * Coordinate tests
 */
bool test_coord_basic(){
  uint N = 100;
  Coords coords = Coords(randi<umat>(N,3,distr_param(0,10)));
  assert(N == coords.num_coords());
  assert(0 == coords.num_special());
  cout << "Finished test_coords_basic" << endl;
}

bool test_coord_basic(){
  Coords coords = Coords(umat{{0,4}, {5,4}, {1,3}});
  TypeRegistry reg = coords._find_oob(uvec{4,4});
  
  
  assert(2 == coords.num_coords());
  assert(0 == coords.num_special());

  
  
  cout << "Finished test_coords_basic" << endl;
}



int main(){
  cout << "Tests start..." << endl;
  test_stride_2d();
  test_stride_3d();
  test_shift_2d();
  test_shift_3d();

  test_coord_basic();
}