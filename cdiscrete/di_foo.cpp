#include "di.h"
#include "misc.h"
#include "io.h"
#include "lcp.h"
#include "solver.h"
#include "basis.h"
#include "smooth.h"
#include "refine.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace tri_mesh;

#define B 5.0
#define LENGTH 0.4
#define GAMMA 0.995
#define SMOOTH_BW 1e9
#define SMOOTH_THRESH 1e-3

#define RBF_GRID_SIZE 4
#define RBF_BW 0.25

#define NUM_REFINE_ITER 2

mat value_freebie(const vec & v,
		  const vector<sp_mat> & blocks,
		  const sp_mat & smoother){
  // If we want a particular function to be well represented in
  // the flow basis, then this function figures out what vectors need to be
  // in the value basis for that function to be contained within all freebie
  // flow bases.
  uint N = v.n_elem;
  assert(size(N,N) == size(smoother));
  uint A = blocks.size();
  assert(A > 0);
  assert(size(N,N) == size(blocks.at(0)));
  
  mat basis = mat(N,A);
  for(uint i = 0; i < A; i++){
    sp_mat E = blocks.at(i) * smoother.t();
    sp_mat G = E.t() + 1e-9*speye(N,N); // Regularized
    basis.col(i) = spsolve(G,v);
  }
  return basis;
}

mat make_raw_value_basis(const Points & points,
                     const vector<sp_mat> & blocks,
                     const sp_mat & smoother){
  uint N = points.n_rows;
  uint A = blocks.size();

  // General basis
  vector<vec> grids;
  grids.push_back(linspace<vec>(-B,B,4));
  grids.push_back(linspace<vec>(-B,B,4));
  Points grid_points = make_points(grids);
  mat grid_basis = make_rbf_basis(points,grid_points,0.27,1e-5);

  // Flow targeted basis
  //mat added_basis = mat(N,5);  
  //mat basis = join_horiz(grid_basis,added_basis);
  mat basis = grid_basis;
    
  return basis; // Don't normalize here
}

mat build_bbox(){
  mat bbox = mat(2,2);
  bbox.col(0).fill(-B);
  bbox.col(1).fill(B);
  return bbox;
}

TriMesh generate_initial_mesh(){
  double angle = 0.125;
  double length = LENGTH;
  mat bbox = build_bbox();
  return generate_initial_mesh(angle,length,bbox);
}

DoubleIntegratorSimulator build_di_simulator(){
  mat bbox = build_bbox();
  mat actions = vec{-1,1};
  double noise_std = 0.0;
  double step = 0.01;
  return DoubleIntegratorSimulator(bbox,actions,noise_std,step);
}

////////////////////
// MAIN FUNCTION //
///////////////////

int main(int argc, char** argv)
{
  arma_rng::set_seed_random();
  // Set up 2D space

  cout << "Generating initial mesh..." << endl;
  TriMesh mesh = generate_initial_mesh();
  Points points = mesh.get_spatial_nodes();
  uint N = points.n_rows;
  assert(N == mesh.number_of_spatial_nodes());
  assert(N > 0);
  
  DoubleIntegratorSimulator di = build_di_simulator();
  uint A = di.num_actions();
  assert(A >= 2);
  
  // Reference blocks
  cout << "Building LCP blocks..." << endl;
  vector<sp_mat> blocks = di.lcp_blocks(&mesh,GAMMA);
  vector<sp_mat> p_blocks = di.transition_blocks(&mesh);
  assert(A == blocks.size());
  assert(size(N,N) == size(blocks.at(0)));
  
  // Build smoother
  cout << "Building smoother matrix..." << endl;
  sp_mat smoother;
  if(SMOOTH_BW < 1e8){
    smoother = gaussian_smoother(points,SMOOTH_BW,SMOOTH_THRESH);
  }
  else{
    smoother = speye(N,N);
  }      
  assert(size(N,N) == size(smoother));

  // Build and pertrub the q
  cout << "Building RHS Q..." << endl;
  mat Q = di.q_mat(&mesh);
  mat costs = Q.tail_cols(A);
  
  bvec free_vars = zeros<bvec>((A+1)*N);
  free_vars.head(N).fill(1);

  sp_mat M = build_M(blocks);
  vec q = vectorise(Q);

  LCP lcp = LCP(M,q,free_vars);

  // Build the approximate PLCP
  mat raw_value_basis = make_raw_value_basis(points,blocks,smoother);
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

  
  PLCP plcp = approx_lcp(value_basis,smoother,blocks,Q,free_vars);
  
  cout << "Projective solve..." << endl;
  SolverResult sol = solver.aug_solve(plcp);
  
  mat P = reshape(sol.p,N,A+1);
  mat D = reshape(sol.d,N,A+1);
  vec V = P.col(0);
  mat F = P.tail_cols(A);
  
  vec res = bellman_residual_at_nodes(&mesh,&di,V,GAMMA);
  vec adv =  advantage_function(&mesh,&di,V,GAMMA);
  uvec p_agg =  policy_agg(&mesh,&di,V,F,GAMMA);

  uvec f_policy = index_min(F,1);
  assert(N == f_policy.n_elem);
  sp_mat Pmc = build_markov_chain_from_blocks(blocks,f_policy);
  vec new_vec = spsolve(Pmc.t(),res);
  
  raw_value_basis = join_horiz(raw_value_basis,new_vec);
  value_basis = sp_mat(orth(raw_value_basis));
  plcp = approx_lcp(value_basis,smoother,blocks,Q,free_vars);
  sol = solver.aug_solve(plcp);

  P = reshape(sol.p,N,A+1);
  D = reshape(sol.d,N,A+1);
  V = P.col(0);
  F = P.tail_cols(A);
  
  res = bellman_residual_at_nodes(&mesh,&di,V,GAMMA);
  adv =  advantage_function(&mesh,&di,V,GAMMA);
  p_agg =  policy_agg(&mesh,&di,V,F,GAMMA);
  
  mesh.write_cgal("test.mesh");
  Archiver arch = Archiver();
  arch.add_mat("ref_P",ref_P);
  arch.add_mat("ref_D",ref_D);
  
  arch.add_mat("P",P);
  arch.add_mat("D",D);

  arch.add_vec("res",res);

  arch.add_vec("adv",adv);
  arch.add_uvec("p_agg",p_agg);

  arch.add_vec("new_vec",new_vec);
  
  arch.write("test.data");
}
