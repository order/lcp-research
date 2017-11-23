#include <armadillo>

#include "grid.h"
#include "io.h"
#include "lcp.h"
#include "basis.h"
#include "misc.h"
#include "plane.h"
#include "points.h"
#include "solver.h"

using namespace arma;
using namespace std;

#define GAMMA 0.997
#define SIM_STEP 0.1
#define NOISE_STD 0.0
#define NMAC_RADIUS 0.25

#define N_XY_GRID_NODES 16
#define N_T_GRID_NODES 8
#define N_OOB_NODES 1
#define N_SAMPLES 1
#define B 1

#define COMP_THRESH 1e-8

mat build_bbox(){
  return mat {{-B,B},{-B,B},{-datum::pi, datum::pi}};
}

RelativePlanesSimulator build_simulator(){
  mat bbox = build_bbox();
  mat actions = mat{{1,0},{0,0},{-1,0}};
  return RelativePlanesSimulator(bbox,
				 actions,
				 NOISE_STD,
				 SIM_STEP,
				 NMAC_RADIUS);
}

////////////////////
// MAIN FUNCTION //
///////////////////

void value_iter_solve(const vector<sp_mat> & p_blocks, const mat & costs){
  cout << "Solving with value iteration..." << endl;
  ValueIteration solver = ValueIteration();
  solver.change_thresh = COMP_THRESH;
  solver.max_iter = 1e8;
  solver.verbose = true;
  
  vec v = solver.solve(p_blocks, GAMMA, costs); 
  
  Archiver arch = Archiver();
  arch.add_vec("v", v);
  arch.add_mat("costs", costs);
  arch.write("/home/epz/scratch/plane_value_iter.data");
}


int main(int argc, char** argv)
{
  arma_rng::set_seed_random();
  
  // Set up 3D space
  mat bbox = build_bbox();
  uvec grid_dim = {N_XY_GRID_NODES, N_XY_GRID_NODES, N_T_GRID_NODES};
  UniformGrid grid = UniformGrid(bbox,
				 grid_dim,
				 N_OOB_NODES);
  mat oob_bbox = {
    {-B,B},
    {-B,B},
    {-datum::inf, arma::datum::inf}
  };
  mat angle_bbox = {
    {-datum::inf, arma::datum::inf},
    {-datum::inf, arma::datum::inf},
    {-datum::pi, datum::pi}
  };

  grid.m_rule_list.emplace_back(new OutOfBoundsRule(oob_bbox,1));
  grid.m_remap_list.emplace_back(new WrapRemapper(angle_bbox));

  cout << "Spatial nodes: " << grid.number_of_spatial_nodes() << endl;
  cout << "Total nodes: " << grid.number_of_all_nodes() << endl;
  
  TypedPoints points = grid.get_all_nodes();
  uint N = points.n_rows;
  cout << "Generated " << N << " nodes" << endl;
  assert(N == grid.number_of_all_nodes());
  assert(N > 0);
  
  RelativePlanesSimulator sim = build_simulator();
  uint A = sim.num_actions();
  assert(A >= 2);

  
  // Build Q matrix
  cout << "Building RHS Q..." << endl;
  mat Q = sim.q_mat(&grid);
  assert(size(N,A+1) == size(Q));

  cout << "Building transition blocks..." << endl;
  vector<sp_mat> p_blocks = sim.transition_blocks(&grid, N_SAMPLES);
  mat costs = Q.tail_cols(A);

  value_iter_solve(p_blocks, costs);


}
