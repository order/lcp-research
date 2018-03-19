#include <assert.h>

#include <armadillo>
#include "basis.h"
#include "grid.h"

using namespace std;
using namespace arma;

/*
 * Test functions for basis functions.
 * Including MultiLinearVarResBasis, the variable resolution multi-linear interpolation basis
 */

sp_mat build_shift_block(uint N, int shift, double p){
  /*
   * Build the N-state hallway transition matrix on a 1D "torus"
   * Stays in the same state with probability p, otherwise transitions
   * to the state "shift" states away.
   */
  assert(0 <= p);
  assert(p <= 1);
  assert(abs(shift) < N);

  vec shift_vec = (1 - p) * ones(N - abs(shift));
  vec wrap_vec = (1 - p) * ones(abs(shift));

  sp_mat shift_diag = spdiag(ones(N - abs(shift)), shift);
  assert(size(N,N) == size(shift_diag));
  sp_mat mod_diag = spdiag(ones(abs(shift)), abs(shift) - N);
  assert(size(N,N) == size(mod_diag));
  
  return p * speye(N,N) + (1 - p) * shift_diag + (1 - p) * mod_diag;
}

bool in_span(const sp_mat & S, const sp_mat & b, double abstol=1e-9){
  /*
   * Check to make sure that b is in the span of sparse matrix S
   * NB: does this by converting to dense; not performant.
   */
  assert(abstol > 0);
  mat A = mat(S);
  mat B = mat(b);

  mat X = solve(A,B);
  mat R = A * X;
  return norm(R - B) < abstol;
}

void test_build_shift_1(){
  /*
   * Building the "shift" matrix
   */
  cout << "test_build_shift_1...";
  cout.flush();
  uint N = 10;
  sp_mat shift = build_shift_block(N, 1, 0.1);

  // Check that it is a probability matrix.
  assert(abs(N - accu(shift)) < 1e-9);
  for(uint i = 0; i < N; i++){
    assert(abs(1 - accu(shift.col(i))) < 1e-9);
  }
  cout << " PASSED." << endl;
}

void test_balance_basis_1(){
  /*
   * Check the ability to balance the simple hallway process
   */
  cout << "test_balance_basis_1...";
  uint N = 5;

  // Build the Lagrangian of the Hallway process
  sp_mat shift = speye(N,N) - 0.95 * build_shift_block(N, 1, 0.1);
  vector<sp_mat> blocks;
  blocks.push_back(shift);

  vector<sp_mat> init;
  sp_mat val = sp_mat(N,1);
  val(0,0) = 1;
  init.push_back(val);  // Add an elementary vectors as value
  init.push_back(sp_mat(N,0)); // Add an empty basis as flow
  
  vector<sp_mat> bal = balance_bases(init, blocks);
  assert(size(N,1) == size(bal.at(0)));
  assert(norm(val - bal.at(0)) < 1e-9);  // Same as input
  
  assert(size(N,1) == size(bal.at(1)));
  cout << " PASSED." << endl;
}

void test_balance_basis_2(){
  cout << "test_balance_basis_2...";
  uint N = 6;
  
  sp_mat shift = speye(N, N) - 0.9 * build_shift_block(N, 1, 0.1);
  vector<sp_mat> blocks;
  blocks.push_back(shift);
  
  vector<sp_mat> init;
  sp_mat flow = sp_mat(N,1);
  flow(0,0) = 1;
  init.push_back(sp_mat(N,0));
  init.push_back(flow);
  
  vector<sp_mat> bal = balance_bases(init, blocks);
  assert(size(N,1) == size(bal.at(0)));
  
  assert(size(N,1) == size(bal.at(1)));
  assert(norm(flow - bal.at(1)) < 1e-9);  // Same as input

  cout << " PASSED." << endl;
}

void test_balance_basis_3(){
  cout << "test_balance_basis_3...";
  uint N = 6;
  
  sp_mat shift = speye(N, N) - 0.9 * build_shift_block(N, 1, 0.1);
  vector<sp_mat> blocks;
  blocks.push_back(shift);
  
  vector<sp_mat> init;
  sp_mat flow = sp_mat(N,1);
  flow(0,0) = 1;
  sp_mat val = sp_mat(N,1);
  val(3,0) = 1;
  init.push_back(val);
  init.push_back(flow);
  
  vector<sp_mat> bal = balance_bases(init, blocks);
  
  assert(size(N,2) == size(bal.at(0)));
  assert(in_span(bal.at(0), val));
  assert(size(N,2) == size(bal.at(1)));
  assert(in_span(bal.at(1), flow));

  cout << " PASSED." << endl;
}

void test_grid_basis_1(){
  cout << "test_grid_basis_1...";
  mat raw_points = mat{ {0, 0}, {0, 1}, {1, 0}, {1, 1}};
  mat bounds = {{0,1}, {0,1}};
 
  TypedPoints points = TypedPoints(raw_points);

  TabularVarResBasis basis_factory = TabularVarResBasis(points, bounds);
  sp_mat basis_mat = basis_factory.get_basis();
  assert(size(4,1) == size(basis_mat));
  assert(abs(basis_mat(0,0) - 0.25) < 1e-9);
  cout << " PASSED." << endl;
}

void test_grid_basis_2(){
  cout << "test_grid_basis_2...";

  mat raw_points = mat{ {0, 0}, {0, 1}, {1, 0}, {1, 1}};
  mat bounds = {{0,1}, {0,1}};
 
  TypedPoints points = TypedPoints(raw_points);

  TabularVarResBasis basis_factory = TabularVarResBasis(points, bounds);
  assert(basis_factory.can_split(0,0));
  basis_factory.split_basis(0,0);
  assert(!basis_factory.can_split(0,0));
  assert(basis_factory.can_split(0,1));

  sp_mat basis_mat = basis_factory.get_basis();
  assert(size(4,2) == size(basis_mat));
  assert(abs(basis_mat(0,0) - 0.5) < 1e-9);
  assert(abs(basis_mat(1,0) - 0.5) < 1e-9);
  assert(abs(basis_mat(2,1) - 0.5) < 1e-9);
  assert(abs(basis_mat(3,1) - 0.5) < 1e-9);
  cout << " PASSED." << endl;

}

void test_grid_basis_3(){
  cout << "test_grid_basis_3...";

  mat raw_points = mat{ {0, 0}, {0, 1}, {1, 0}, {1, 1}};
  mat bounds = {{0,1}, {0,1}};
 
  TypedPoints points = TypedPoints(raw_points);

  TabularVarResBasis basis_factory = TabularVarResBasis(points, bounds);
  basis_factory.split_basis(0,0);
  basis_factory.split_basis(0,1);

  sp_mat basis_mat = basis_factory.get_basis();
  assert(size(4,3) == size(basis_mat));
  assert(abs(basis_mat(0,0) - 1) < 1e-9);
  assert(abs(basis_mat(1,2) - 1) < 1e-9);
  assert(abs(basis_mat(2,1) - 0.5) < 1e-9);
  assert(abs(basis_mat(3,1) - 0.5) < 1e-9);
  cout << " PASSED." << endl;

}


void test_interp_grid_basis_1(){
  cout << "test_interp_grid_basis_1...";
  uvec grid_size = {2,2};
  MultiLinearVarResBasis basis_factory = MultiLinearVarResBasis(grid_size);
  
  sp_mat basis = basis_factory.get_basis();
  assert(5 == basis.n_rows);
  assert(5 == basis.n_cols);
  cout << " PASSED." << endl;

}

void test_interp_grid_basis_2(){
  cout << "test_interp_grid_basis_2...";
  
  uvec grid_size = {3,3};
  MultiLinearVarResBasis basis_factory = MultiLinearVarResBasis(grid_size);
  
  mat basis = mat(basis_factory.get_basis());
  assert(10 == basis.n_rows);
  assert(5 == basis.n_cols);

  // Rows sum to 1
  assert(abs(accu(sum(basis,1) - 1)) < PRETTY_SMALL);

  
  umat vertices = box2vert(umat{{0,2},{0,2}}, grid_size); 
  uvec vidx = coords_to_indices(grid_size,
				Coords(conv_to<imat>::from(vertices)));

  for(uint i = 0; i < vidx.n_elem; i++){
    // Rows for vertices are pure
    
    assert(abs(max(basis.row(vidx(i))) - 1) < PRETTY_SMALL);
  }
  
  cout << " PASSED." << endl;
}

void test_interp_grid_basis_3(){
  cout << "test_interp_grid_basis_3...";
  
  uvec grid_size = {4,4};
  MultiLinearVarResBasis basis_factory = MultiLinearVarResBasis(grid_size);
  assert(basis_factory.can_split(0,0));

  // Split into two
  uint new_cell = basis_factory.split_cell(0,0);
  assert(1 == new_cell);

  // Expected number of points: 17 (4 x 4 + 1)
  // Expected number of bases: 7 (6 + 1)
  // o - o - o
  // |   |   |
  // o - o - o   (*)
  mat basis = mat(basis_factory.get_basis());
  assert(size(17,7) == size(basis));

  cout << " PASSED." << endl;
}

void test_interp_grid_basis_4(){
  cout << "test_interp_grid_basis_4...";

  uvec grid_size = {4,4};
  MultiLinearVarResBasis basis_factory = MultiLinearVarResBasis(grid_size);
  basis_factory.split_per_dimension(0, uvec{1,1});

  assert(4 == basis_factory.m_cell_to_bbox.size()); // 4 cells
  mat basis = mat(basis_factory.get_basis());
  assert(size(17,10) == size(basis));
  cout << " PASSED." << endl;

}


int main(){
  test_build_shift_1();
  
  test_balance_basis_1();
  test_balance_basis_2();
  test_balance_basis_3();

  test_grid_basis_1();
  test_grid_basis_2();
  test_grid_basis_3();

  test_interp_grid_basis_1();
  test_interp_grid_basis_2();
  test_interp_grid_basis_3();
  test_interp_grid_basis_4();
}
