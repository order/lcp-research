#include <armadillo>

#include "grid.h"
#include "io.h"
#include "lcp.h"
#include "basis.h"
#include "misc.h"
#include "plane.h"
#include "points.h"
#include "refine.h"
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
#define N_SAMPLES 5
#define B 1
#define IGNORE_Q true

#define COMP_THRESH 1e-8
#define NUM_ADD_ROUNDS 4

#define USE_OOB_NODE


#define BASIS_G 3
#define BASIS_BW 1.0

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

  //mat dense_basis = make_rbf_basis(points, grid_points, BASIS_BW);
  mat dense_basis = mat(make_voronoi_basis(points, grid_points));
  return dense_basis;
}

RelativePlanesSimulator build_simulator(){
  mat bbox = build_bbox();
  mat actions = mat{{1,0},{0,0},{-1,0}};
  double noise_std = NOISE_STD;
  double step = SIM_STEP;
  double nmac_radius = NMAC_RADIUS;
  return RelativePlanesSimulator(bbox, actions, noise_std, step, nmac_radius);
}

vec find_residual(const UniformGrid & grid,
                  const RelativePlanesSimulator & sim,
                  const mat & P){
  vec V = P.col(0);
  return bellman_residual_at_nodes(&grid, &sim, V, GAMMA);
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
  psolver.comp_thresh = COMP_THRESH;
  psolver.initial_sigma = 0.25;
  psolver.verbose = true;
  psolver.iter_verbose = true;
  psolver.regularizer = 0;

  
  PLCP plcp = approx_lcp(sp_mat(basis),smoother,
                         blocks,Q,free_vars, IGNORE_Q);
  sp_mat M = build_M(blocks);
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
  //grid.m_remap_list.emplace_back(new SaturateRemapper(oob_bbox));
  grid.m_remap_list.emplace_back(new WrapRemapper(angle_bbox));

  cout << "Spatial nodes: " << grid.number_of_spatial_nodes() << endl;
  cout << "Total nodes: " << grid.number_of_all_nodes() << endl;


  
  TypedPoints points = grid.get_all_nodes();
  uint N = points.n_rows;
  cout << "Generated " << N << " spatial nodes" << endl;
  assert(N == grid.number_of_all_nodes());
  assert(N == grid.number_of_spatial_nodes() + 1);
  assert(N > 0);
  
  RelativePlanesSimulator sim = build_simulator();
  uint A = sim.num_actions();
  assert(A >= 2);
  
  // Reference blocks
  cout << "Building LCP blocks..." << endl;
  vector<sp_mat> blocks = sim.lcp_blocks(&grid, GAMMA, N_SAMPLES);
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
  bvec free_vars = zeros<bvec>((A+1) * N);
  free_vars.head(N).fill(1);

  cout << "Making value basis..." << endl;
  mat dense_basis = make_basis(points);
  //mat dense_basis = eye(N,N);  // AHHHH UNDO
  sp_mat basis = sp_mat(dense_basis);

  cube primals = cube(N,A+1,NUM_ADD_ROUNDS+1);
  cube duals = cube(N,A+1,NUM_ADD_ROUNDS+1);
  cube added_bases = cube(N,A,NUM_ADD_ROUNDS);
  mat residuals = mat(N,NUM_ADD_ROUNDS);

  for(uint I = 0; I < NUM_ADD_ROUNDS+1; I++){
    cout << "Running " << (I+1) << "/" << (NUM_ADD_ROUNDS+1)  << endl;
    
    cout << "\tFinding a solution..." << endl;
    SolverResult sol = find_solution(basis,
				     smoother,
				     blocks,
				     Q,
				     free_vars);
    mat P = reshape(sol.p, N, A+1);
    mat D = reshape(sol.d, N, A+1);
    primals.slice(I) = P;
    duals.slice(I) = D;

    if(NUM_ADD_ROUNDS == I){
      break; // Final Round
    }

    vec md_res = min(D.tail_cols(A),1);
    vec res = find_residual(grid, sim, P);
    residuals.col(I) = res;
    vec bellman_res_norm = vec{
      norm(res,1),
      norm(res,2),
      norm(res,"inf")};
    vec md_res_norm = vec{
      norm(md_res,1),
      norm(md_res,2),
      norm(md_res,"inf")};

    cout << "\tBellman residual norm:" << bellman_res_norm.t() << endl;
    cout << "\tMin. dual residual norm:" << md_res_norm.t() << endl;
	
    uvec pi = index_max(P.tail_cols(A), 1);
    mat new_bases = mat(N, A);
    for(uint a = 0; a < A; a++){
      vec heur;
      if (norm(md_res,"inf") < 2){
	heur = res;
      }
      else{
	heur = md_res;
      }
      heur = Q.col(1);
      uvec mask = find(a != pi);
      heur(mask).fill(0); // mask out
      cout << "\tForming residual basis element " << a << "..." << endl;
      new_bases.col(a) = spsolve(smoother * blocks.at(a).t(), heur);
    }
    added_bases.slice(I) = new_bases;

    cout << "Orthgonalizing and appending..." << endl;
    dense_basis = join_horiz(dense_basis, new_bases);
    dense_basis = orth(dense_basis);
    basis = sp_mat(dense_basis);
  }
  
  Archiver arch = Archiver();
  arch.add_cube("primals", primals);
  arch.add_cube("duals", duals);
  arch.add_cube("added_bases", added_bases);
  arch.add_mat("residuals", residuals);
  arch.write(DATA_FILE_NAME);
}