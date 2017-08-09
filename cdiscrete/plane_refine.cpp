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
#define N_XY_GRID_NODES 32
#define N_T_GRID_NODES 16
#define N_OOB_NODES 1
#define N_SAMPLES 5
#define B 1


#define BASIS_G 3
#define BASIS_BW 10.0

#define DATA_FILE_NAME "/home/epz/scratch/plane_refine.data"

mat build_bbox(){
  return mat {{-B,B},{-B,B},{-datum::pi, datum::pi}};
}

mat make_basis(const TypedPoints & points){
  uint N = points.n_rows;

  // General basis
  vector<vec> grids;
  grids.push_back(linspace<vec>(-B,B,BASIS_G));
  grids.push_back(linspace<vec>(-B,B,BASIS_G));
  grids.push_back(linspace<vec>(-datum::pi,datum::pi,BASIS_G));
  Points grid_points = make_points<mat,vec>(grids);
  
  mat basis = make_rbf_basis(points, grid_points, BASIS_BW);

  return basis; // Don't normalize here
}

RelativePlanesSimulator build_simulator(){
  mat bbox = build_bbox();
  mat actions = mat{{1,0},{-1,0}};
  double noise_std = 0.1;
  double step = 0.01;
  double nmac_radius = 0.25;
  return RelativePlanesSimulator(bbox, actions, noise_std, step, nmac_radius);
}


SolverResult find_solution(const sp_mat & basis,
                const sp_mat & smoother,
                const vector<sp_mat> & blocks,
                const mat & Q,
                const bvec & free_vars){
  uint N = basis.n_rows;
  assert(size(N,N) == size(smoother));
  uint A = blocks.size();
  assert(size(N,A+1) == size(Q));

  ProjectiveSolver psolver = ProjectiveSolver();
  psolver.comp_thresh = 1e-12;
  psolver.initial_sigma = 0.25;
  psolver.verbose = true;
  psolver.iter_verbose = true;
  psolver.regularizer = 0;
  
  PLCP plcp = approx_lcp(sp_mat(basis),smoother,
                         blocks,Q,free_vars);
  return psolver.aug_solve(plcp);
}

SolverResult find_solution(const mat & basis,
                const sp_mat & smoother,
                const vector<sp_mat> & blocks,
                const mat & Q,
                const bvec & free_vars){
  sp_mat sp_basis = sp_mat(basis);
  return find_solution(sp_basis,
		       smoother,
		       blocks,
		       Q,
		       free_vars);
}

////////////////////
// MAIN FUNCTION //
///////////////////

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


  
  TypedPoints points = grid.get_spatial_nodes();
  uint N = points.n_rows;
  cout << "Generated " << N << " spatial nodes" << endl;
  assert(N == grid.number_of_spatial_nodes());
  assert(N > 0);
  
  RelativePlanesSimulator sim = build_simulator();
  uint A = sim.num_actions();
  assert(A >= 2);
  
  // Reference blocks
  cout << "Building LCP blocks..." << endl;
  vector<sp_mat> blocks = sim.lcp_blocks(&grid, GAMMA,N_SAMPLES);
  assert(A == blocks.size());
  assert(size(N,N) == size(blocks.at(0)));
  
  // Build smoother
  cout << "Building smoother matrix..." << endl;
  sp_mat smoother = speye(N,N);

  // Build Q matrix
  cout << "Building RHS Q..." << endl;
  mat Q = sim.q_mat(&grid);
  cout << "size(Q): " << size(Q) << endl;
  cout << "Expected size(Q): " << size(N,A+1) << endl;

  assert(size(N,A+1) == size(Q));

  // Extract costs
  mat costs = Q.tail_cols(A);

  // Free the cost function from non-neg constraints
  bvec free_vars = zeros<bvec>((A+1)*N);
  free_vars.head(N).fill(1);

  cout << "Making value basis..." << endl;
  mat basis = make_basis(points);

  cout << "Finding a solution..." << endl;
  SolverResult sol = find_solution(basis, smoother, blocks, Q, free_vars);

  
  Archiver arch = Archiver();
  arch.add_vec("p",sol.p);
  arch.add_vec("d",sol.d);
  arch.add_vec("q",vectorise(Q));
  arch.add_mat("basis", basis);
  arch.write(DATA_FILE_NAME);
}
