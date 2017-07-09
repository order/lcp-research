#include <armadillo>
#include <assert.h>

#include "grid.h"

using namespace arma;
using namespace std;

/*
 * Helper function tests
 */
#define ALMOST_EQUAL(A,B) approx_equal(A,B,"both",PRETTY_SMALL,PRETTY_SMALL)

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
  Coords coords = Coords(randi<imat>(N, 3, distr_param(0, 9)));
  assert(N == coords.num_coords());
  assert(0 == coords.num_special());
  cout << "Finished test_coords_basic" << endl;
}

bool test_coord_oob(){
  imat raw_coords = imat{
    {0,4},
    {Coords::SPECIAL_FILL,Coords::SPECIAL_FILL},
    {1,3}};
  Coords coords = Coords(raw_coords);
  assert(3 == coords.num_coords());
  assert(1 == coords.num_special());
  
  TypeRegistry reg = coords._find_oob(uvec{4,4});
  assert(2 == reg.size());  
  cout << "Finished test_coords_oob" << endl;
}

bool test_coord_map_check(){
  /* Checks that the coords -> indices -> coords works as identity*/
  uint N = 100;
  Coords coords = Coords(randi<imat>(N, 3, distr_param(0, 9)));
  uvec grid_size = uvec{10, 10, 10};
  uvec indices = coords.get_indices(grid_size);
  Coords recovered = indices_to_coords(grid_size, indices);
  assert(coords.equals(recovered));
  cout << "Finished test_coord_map_check" << endl;

}

bool test_coord_map_check_2(){
  /* Checks that the coords -> indices -> coords works as identity*/
  uint N = 10;
  TypeRegistry reg;
  reg[3] = 1;
  reg[5] = 6;
  Coords coords = Coords(randi<imat>(N, 3, distr_param(0, 9)), reg);
  assert(2 == coords.num_special());
  
  uvec grid_size = uvec{10, 10, 10};
  uvec indices = coords.get_indices(grid_size);
  Coords recovered = indices_to_coords(grid_size, indices);
  assert(coords.equals(recovered));
  cout << "Finished test_coord_map_check_2" << endl;
}

bool test_grid_basic_1(){
  vec low = vec{0,0};
  vec high = vec{10,10};
  uvec num_cells = uvec{10,10};  
  UniformGrid grid = UniformGrid(low,high,num_cells,1);
  
  TypedPoints p = TypedPoints(mat{{1,1}});
  Coords c = grid.points_to_cell_coords(p);
  assert(1 == c.num_spatial());
  Coords ref = Coords(ones<imat>(1,2));
  assert(ref.equals(c));
  cout << "Finished test_grid_basic_1" << endl;
}

bool test_grid_basic_2(){
  vec low = vec{1,1};
  vec high = vec{11,11};
  uvec num_cells = uvec{10,10};  
  UniformGrid grid = UniformGrid(low,high,num_cells,1);
  
  TypedPoints p = TypedPoints(mat{{1,1}});
  Coords c = grid.points_to_cell_coords(p);
  assert(1 == c.num_spatial());
  Coords ref = Coords(zeros<imat>(1,2));
  assert(ref.equals(c));
  cout << "Finished test_grid_basic_2" << endl;
}

bool test_grid_cell_coords_to_low_points_1(){
  vec low = vec{0,0};
  vec high = vec{10,10};
  uvec num_cells = uvec{10,10};  
  UniformGrid grid = UniformGrid(low,high,num_cells,1);
  
  Coords c = Coords(ones<imat>(1,2));
  TypedPoints p = grid.cell_coords_to_low_points(c);
  assert(ALMOST_EQUAL(ones<mat>(1,2), p.m_points));
  cout << "Finished test_grid_cell_coords_to_low_points_1" << endl;
}

bool test_grid_cell_coords_to_low_points_2(){
  vec low = vec{0,0};
  vec high = vec{10,10};
  uvec num_cells = uvec{10,10};  
  UniformGrid grid = UniformGrid(low,high,num_cells,1);

  TypedPoints p = TypedPoints(11*ones<mat>(1,2));
  OutOfBoundsRule rule = OutOfBoundsRule(grid.find_bounding_box(), 1);
  p.apply_typing_rule(rule);
  
  Coords c = grid.points_to_cell_coords(p);
  assert(c.is_special(0));
  TypedPoints p2 = grid.cell_coords_to_low_points(c);
  assert(p2.is_special(0));
  cout << "Finished test_grid_cell_coords_to_low_points_2" << endl;

}

bool test_grid_cell_coords_to_low_points_3(){
  vec low = vec{0,0};
  vec high = vec{10,10};
  uvec num_cells = uvec{10,10};  
  UniformGrid grid = UniformGrid(low,high,num_cells,1);

  TypedPoints p = TypedPoints(10*ones<mat>(1,2));
  OutOfBoundsRule rule = OutOfBoundsRule(grid.find_bounding_box(), 1);
  p.apply_typing_rule(rule);
  
  Coords c = grid.points_to_cell_coords(p);
  assert(!c.is_special(0));
  TypedPoints p2 = grid.cell_coords_to_low_points(c);
  assert(!p2.is_special(0));
  assert(ALMOST_EQUAL(p.m_points, p2.m_points));
  cout << "Finished test_grid_cell_coords_to_low_points_3" << endl;

}


bool test_grid_points_to_dist_1(){
  // Dead center of the unit cube
  vec low = vec{0,0};
  vec high = vec{1,1};
  uvec num_cells = uvec{1,1};  
  UniformGrid grid = UniformGrid(low,high,num_cells,1);

  TypedPoints p = TypedPoints(mat{{0.5,0.5}});
  mat d = grid.points_to_cell_nodes_dist(p);
  mat ref = (1 / sqrt(2.0)) * ones<mat>(1,4);
  assert(ALMOST_EQUAL(d, ref));

  cout << "Finished test_grid_points_to_dist_1" << endl; 
}

bool test_grid_points_to_dist_2(){
  // As above but shifted
  vec low = vec{1,1};
  vec high = vec{2,2};
  uvec num_cells = uvec{1,1};  
  UniformGrid grid = UniformGrid(low,high,num_cells,1);

  TypedPoints p = TypedPoints(mat{{1.5,1.5}});
  mat d = grid.points_to_cell_nodes_dist(p);
  mat ref = (1 / sqrt(2.0)) * ones<mat>(1,4);
  assert(ALMOST_EQUAL(d, ref));

  cout << "Finished test_grid_points_to_dist_2" << endl; 
}

bool test_grid_points_to_dist_3(){
  // As above but larger grid
  vec low = vec{1,1};
  vec high = vec{3,3};
  uvec num_cells = uvec{2,2};  
  UniformGrid grid = UniformGrid(low,high,num_cells,1);

  TypedPoints p = TypedPoints(mat{{1.5,1.5}});
  mat d = grid.points_to_cell_nodes_dist(p);
  mat ref = (1 / sqrt(2.0)) * ones<mat>(1,4);
  assert(ALMOST_EQUAL(d, ref));

  cout << "Finished test_grid_points_to_dist_3" << endl; 
}

bool test_grid_points_to_dist_4(){
  // Dead center of a 3d unit cube
  vec low = vec{0,0,0};
  vec high = vec{1,1,1};
  uvec num_cells = uvec{1,1,1};
  UniformGrid grid = UniformGrid(low,high,num_cells,1);

  TypedPoints p = TypedPoints(0.5*ones<mat>(1,3));
  mat d = grid.points_to_cell_nodes_dist(p);
  mat ref = (sqrt(3.0 / 4.0)) * ones<mat>(1,8);
  assert(ALMOST_EQUAL(d, ref));

  cout << "Finished test_grid_points_to_dist_4" << endl; 
}

bool test_grid_points_to_dist_5(){
  // Out of bounds
  vec low = vec{0,0,0};
  vec high = vec{1,1,1};
  uvec num_cells = uvec{1,1,1};
  UniformGrid grid = UniformGrid(low,high,num_cells,1);
  grid.m_rule_list.emplace_back(new OutOfBoundsRule(grid.find_bounding_box(),
						    1));

  TypedPoints p = TypedPoints(2*ones<mat>(1,3));
  mat d = grid.points_to_cell_nodes_dist(p);
  mat ref = zeros<mat>(1,8);
  ref(0) = 1.0;
  assert(ALMOST_EQUAL(d, ref));

  cout << "Finished test_grid_points_to_dist_5" << endl; 
}

bool test_element_dist_1(){
  vec low = vec{0,0};
  vec high = vec{1,1};
  uvec num_cells = uvec{1,1};  
  UniformGrid grid = UniformGrid(low,high,num_cells,1);

  
}


int main(){
  cout << "Tests start..." << endl;
  // Stride and shift testing
  test_stride_2d();
  test_stride_3d();
  test_shift_2d();
  test_shift_3d();

  // Coord object testing
  test_coord_basic();
  test_coord_oob();
  test_coord_map_check();
  test_coord_map_check_2();

  // Grid testing.
  test_grid_basic_1();
  test_grid_basic_2();
  test_grid_cell_coords_to_low_points_1();
  test_grid_cell_coords_to_low_points_2();
  test_grid_cell_coords_to_low_points_3();
  
  test_grid_points_to_dist_1();
  test_grid_points_to_dist_2();
  test_grid_points_to_dist_3();
  test_grid_points_to_dist_4();
  test_grid_points_to_dist_5();
}
