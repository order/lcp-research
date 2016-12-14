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
#define LENGTH 0.35
#define GAMMA 0.995
#define SMOOTH_BW 1e9
#define SMOOTH_THRESH 1e-3

vec new_vector(const Points & points){
  double angle = 1.2567;
  double b = 0.001;
  double c = 0.0785;
  mat rot = mat{{cos(angle), -sin(angle)},{sin(angle),cos(angle)}};
  mat cov = rot * diagmat(vec{b,c}) * rot.t();
  return gaussian(points,zeros<vec>(2),cov);
}

mat make_raw_value_basis(const Points & points,
                     const vector<sp_mat> & blocks,
                     const sp_mat & smoother){
  uint N = points.n_rows;
  uint A = blocks.size();

  // General basis
  vector<vec> grids;
  grids.push_back(linspace<vec>(-B,B,3));
  grids.push_back(linspace<vec>(-B,B,3));
  Points grid_points = make_points(grids);
  mat basis = make_rbf_basis(points,grid_points,0.25,1e-5);
  
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
  
  PLCP plcp = approx_lcp(value_basis,smoother,blocks,Q,free_vars);
  
  KojimaSolver solver = KojimaSolver();
  //ProjectiveSolver solver = ProjectiveSolver();
  solver.comp_thresh = 1e-22;
  solver.initial_sigma = 0.2;
  solver.aug_rel_scale = 0.75;
  solver.regularizer = 0;
  solver.verbose = true;

  SolverResult sol = solver.aug_solve(lcp);
  //SolverResult sol = solver.aug_solve(plcp);
  
  mat P = reshape(sol.p,N,A+1);
  mat D = reshape(sol.d,N,A+1);
  vec V = P.col(0);
  mat F = P.tail_cols(A);
  
  vec res = bellman_residual_at_nodes(&mesh,&di,V,GAMMA);
  vec res_faces = bellman_residual_at_centers(&mesh,&di,V,GAMMA);
  vec res_norm = vec{norm(res,1),
		     norm(res,2),
		     norm(res,"inf")};
  vec adv =  advantage_function(&mesh,&di,V,GAMMA);
  
  mesh.write_cgal("test.mesh");
  Archiver arch = Archiver();
  arch.add_mat("primal",P);
  arch.add_mat("dual",D);
  arch.add_vec("residual",res);

  
  arch.write("test.data");
}
