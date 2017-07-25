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
#define N_GRID_NODES 12
#define N_OOB_NODES 1
#define B 1

#define DATA_FILE_NAME "/home/epz/scratch/test.data"

mat build_bbox(){
  return mat {{-B,B},{-B,B},{-datum::pi, datum::pi}};
}

RelativePlanesSimulator build_simulator(){
  mat bbox = build_bbox();
  mat actions = mat{{0,0},{1,0},{-1,0}};
  double noise_std = 0.1;
  double step = 0.01;
  double nmac_radius = 0.25;
  return RelativePlanesSimulator(bbox, actions, noise_std, step, nmac_radius);
}

////////////////////
// MAIN FUNCTION //
///////////////////

int main(int argc, char** argv)
{
  arma_rng::set_seed_random();
  
  // Set up 3D space
  mat bbox = build_bbox();
  UniformGrid grid = UniformGrid(bbox,
				 N_GRID_NODES * ones<uvec>(THREE_DIM),
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
  vector<sp_mat> blocks = sim.lcp_blocks(&grid, GAMMA);
  vector<sp_mat> p_blocks = sim.transition_blocks(&grid);
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

  // Build the LCP
  sp_mat M = build_M(blocks);
  cout << "Generated LCP matrix: " << size(M) << endl;
  vec q = vectorise(Q);
  LCP lcp = LCP(M,q,free_vars);

  
  KojimaSolver ref_solver = KojimaSolver();
  ref_solver.comp_thresh = 1e-6;
  ref_solver.initial_sigma = 0.2;
  ref_solver.aug_rel_scale = 1.5;
  ref_solver.regularizer = 1e-9;
  ref_solver.verbose = true;
  ref_solver.iter_verbose = true;
  ref_solver.save_system = true;
  

  cout << "Reference solve..." << endl;
  SolverResult ref_sol = ref_solver.aug_solve(lcp);
  mat ref_P = reshape(ref_sol.p,N,A+1);
  mat ref_D = reshape(ref_sol.d,N,A+1);
  
  Archiver arch = Archiver();
  arch.add_mat("ref_P",ref_P);
  arch.add_mat("ref_D",ref_D);
  arch.add_sp_mat("M",M);
  arch.add_mat("Q",Q);
  arch.write(DATA_FILE_NAME);
}
