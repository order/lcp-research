#include "di.h"
#include "misc.h"
#include "io.h"
#include "lcp.h"
#include "solver.h"
#include <armadillo>
#include "plane.h"

using namespace arma;

#define GAMMA 0.99
#define N_GRID_NODES 50

"""
This is a fast-moving file for double integrator experiments.
It generates:
1) Mesh file
2) Data file
These are read and visualized in ../di_foo_viewer.py
"""
mat build_bbox(){
  return mat {{0,1},{0,1},{-arma::datum::pi, arma::datum::pi}};
}

RelativePlaneSimulator build_simulator(){
  mat bbox = build_bbox();
  mat actions = vec{{0,1},{1,1},{-1,1}};
  double noise_std = 0.0;
  double step = 0.01;
  return RelativePlaneSimulator(bbox,actions,noise_std,step);
}

////////////////////
// MAIN FUNCTION //
///////////////////

int main(int argc, char** argv)
{
  arma_rng::set_seed_random();
  
  // Set up 3D space
  cout << "Generating initial mesh..." << endl;
  mat bbox = build_bbox();
  UniformGrid grid = UniformGrid(bbox.col(0) bbox.col(1), N_GRID_NODES);
  Points points = grid.get_spatial_nodes();
  uint N = points.n_rows;
  assert(N == mesh.number_of_spatial_nodes());
  assert(N > 0);
  
  RelativePlanesSimulator sim = build_simulator();
  uint A = di.num_actions();
  assert(A >= 2);
  
  // Reference blocks
  cout << "Building LCP blocks..." << endl;
  vector<sp_mat> blocks = sim.lcp_blocks(&grid,GAMMA);
  vector<sp_mat> p_blocks = sim.transition_blocks(&grid);
  assert(A == blocks.size());
  assert(size(N,N) == size(blocks.at(0)));
  
  // Build smoother
  cout << "Building smoother matrix..." << endl;
  sp_mat smoother = speye(N,N);

  // Build and pertrub the q
  cout << "Building RHS Q..." << endl;
  mat Q = sim.q_mat(&grid);
  mat costs = Q.tail_cols(A);
  
  bvec free_vars = zeros<bvec>((A+1)*N);
  free_vars.head(N).fill(1);

  sp_mat M = build_M(blocks);
  vec q = vectorise(Q);

  LCP lcp = LCP(M,q,free_vars);

  // Build the approximate PLCP
  mat raw_value_basis = make_raw_value_basis(points, blocks, smoother);
  sp_mat value_basis = sp_mat(orth(raw_value_basis));
  uint K = value_basis.n_cols;

  // Record raw (un-orthonormalized) bases
  vector<mat> raw_flow_bases = make_raw_freebie_flow_bases(raw_value_basis,
                                                           blocks,Q);
  assert(A == raw_flow_bases.size());
  cube basis_cube = cube(N,K+1,A+1);
  basis_cube.slice(0) = join_horiz(raw_value_basis,zeros<mat>(N,1));
  for(uint i = 0; i < A; i++){
    // Extra column in flow bases for cost column
    assert(size(N,K+1) == size(raw_flow_bases.at(i)));
    basis_cube.slice(i+1) = raw_flow_bases.at(i);
  }

  
  KojimaSolver ref_solver = KojimaSolver();
  ref_solver.comp_thresh = 1e-22;
  ref_solver.initial_sigma = 0.2;
  ref_solver.aug_rel_scale = 0.75;
  ref_solver.regularizer = 1e-12;
  ref_solver.verbose = false;
  ref_solver.iter_verbose = false;
  

  cout << "Reference solve..." << endl;
  SolverResult ref_sol = ref_solver.aug_solve(lcp);
  mat ref_P = reshape(ref_sol.p,N,A+1);
  mat ref_D = reshape(ref_sol.d,N,A+1);

  ProjectiveSolver solver = ProjectiveSolver();
  solver.comp_thresh = 1e-22;
  solver.initial_sigma = 0.2;
  solver.aug_rel_scale = 0.75;
  solver.regularizer = 1e-12;
  solver.verbose = false;
  solver.iter_verbose = false;

  ////////////////////////////////////////
  // Get residual from initial approximation
  PLCP plcp = approx_lcp(value_basis,smoother,blocks,Q,free_vars);
  cout << "Projective solve..." << endl;
  SolverResult sol = solver.aug_solve(plcp);

  // Add new basis
  mat old_P = reshape(sol.p,N,A+1);
  vec old_res = bellman_residual_at_nodes(&mesh,&di,old_P.col(0),GAMMA);
  vec value_res = blocks.at(0) * old_P.col(1) + blocks.at(1) * old_P.col(2) - Q.col(0);
  
  mat dual_res = reshape(sol.d,N,A+1).eval().tail_cols(A);
  
  mat new_vects = mat(N,2);
  for(uint i = 0; i < A; i++){
    vec target = min(dual_res,1);
    uvec mask = find((1-i) == index_min(dual_res,1));
    target(mask).fill(0); // mask out

    new_vects.col(i) = spsolve(blocks.at(i).t(),target);
    }
  
  raw_value_basis = join_horiz(raw_value_basis,new_vects);
  value_basis = sp_mat(orth(raw_value_basis));

  // Re-solve
  plcp = approx_lcp(value_basis,smoother,blocks,Q,free_vars);
  sol = solver.aug_solve(plcp);

  // Report outcome.
  mat P = reshape(sol.p,N,A+1);
  mat D = reshape(sol.d,N,A+1);
  vec V = P.col(0);
  mat F = P.tail_cols(A);
  
  vec res = bellman_residual_at_nodes(&mesh,&di,V,GAMMA);
  vec adv =  advantage_function(&mesh,&di,V,GAMMA);
  uvec p_agg =  policy_agg(&mesh,&di,V,F,GAMMA);

  vec res_norm = vec{norm(res,1),norm(res,2),norm(res,"inf")};
  cout << "Residual norm:" << res_norm.t() << endl;
  
  mesh.write_cgal(MESH_FILE_NAME);
  Archiver arch = Archiver();
  arch.add_mat("ref_P",ref_P);
  arch.add_mat("ref_D",ref_D);

  arch.add_mat("P",P);
  arch.add_mat("D",D);

  arch.add_vec("old_res",old_res);
  arch.add_vec("res",res);
  arch.add_vec("value_res",value_res);

  arch.add_vec("adv",adv);
  arch.add_uvec("p_agg",p_agg);

  arch.add_mat("new_vects",new_vects);
  
  arch.write(DATA_FILE_NAME);
}
