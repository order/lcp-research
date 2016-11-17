#include "di.h"
#include "misc.h"
#include "io.h"
#include "lcp.h"
#include "solver.h"
#include "basis.h"
#include "smooth.h"
#include "refine.h"

/*
  Program for exploring the effects of adding different isometric gaussians
  to the origin for approximating a double integrator problem.
  Record l1,l2,linf Bellmen residual changes
*/

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace tri_mesh;

#define B 5.0
#define LENGTH 0.35
#define GAMMA 0.995
#define SMOOTH_BW 1e9
#define SMOOTH_THRESH 1e-4

#define RBF_GRID_SIZE 8 // Make sure even so origin isn't already covered

sp_mat make_value_basis(const Points & points){

  uint N = points.n_rows;
  
  uint k = RBF_GRID_SIZE;
  vec grid = linspace<vec>(-B,B,k);
  vector<vec> grids;
  grids.push_back(grid);
  grids.push_back(grid);

  mat centers = make_points(grids);
  double bandwidth = 0.75;
  mat basis = make_rbf_basis(points,centers,bandwidth,1e-6);
  //sp_mat basis = make_voronoi_basis(points,centers);
  //sp_mat basis = speye(N,N);
  return sp_mat(basis);
}

mat refine(const Points & vertices,
           const Points & face_centers,
           const vec & heuristic){
  cout << "Starting refine..." << endl;
  uint F = face_centers.n_rows;
  uint N = vertices.n_rows;
  uint D = vertices.n_cols;
  assert(F == heuristic.n_elem);
  
  uint cand = 256;
  uvec center_idx = randi<uvec>(cand,distr_param(0,F-1));
  double bw = 5;

  mat basis = mat(F,cand);
  for(uint i = 0; i < cand; i++){
    uint idx = center_idx(i);
    assert(idx < F);
    vec center = face_centers.row(idx).t();
    basis.col(i) = gaussian(face_centers,center,bw);
  }
  //basis = orth(basis);
  vec corr = basis.t() * heuristic;
  uvec corr_idx = sort_index(corr);

  uint K = 1;
  mat actual_basis = mat(N,K);
  uvec actual_idx = corr_idx.tail(K);

  for(uint k = 0; k < K; k++){
    uint I = actual_idx(k);

    uint idx = center_idx(I);
    vec center = face_centers.row(idx).t();
    actual_basis.col(k) = gaussian(vertices,center,bw);
  }
  return actual_basis;
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
  double step = 0.025;
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
  assert(A == blocks.size());
  assert(size(N,N) == size(blocks.at(0)));

  // Build smoother
  cout << "Building smoother matrix..." << endl;
  sp_mat smoother = gaussian_smoother(points,SMOOTH_BW,SMOOTH_THRESH);
  assert(size(N,N) == size(smoother));

  // Build and pertrub the q
  cout << "Building RHS Q..." << endl;
  mat Q = di.q_mat(&mesh);
  
  bvec free_vars = zeros<bvec>((A+1)*N);
  free_vars.head(N).fill(1);

  cout << "Making value basis..." << endl;
  sp_mat value_basis = make_value_basis(points);


  
  cout << "Building reference approximate PLCP..." << endl;
  PLCP ref_plcp = approx_lcp(value_basis,smoother,blocks,Q,free_vars);

  ProjectiveSolver psolver = ProjectiveSolver();
  psolver.comp_thresh = 1e-8;
  psolver.initial_sigma = 0.25;
  psolver.verbose = false;
  psolver.iter_verbose = false;

  cout << "Starting reference augmented PLCP solve..."  << endl;
  SolverResult ref_sol = psolver.aug_solve(ref_plcp);
  mat ref_P = reshape(ref_sol.p,N,A+1);
  vec ref_V = ref_P.col(0);
  vec ref_res = bellman_residual_at_nodes(&mesh,&di,ref_V,GAMMA);  
  vec ref_res_norm = vec{norm(ref_res,1),norm(ref_res,2),norm(ref_res,"inf")};

  uint BW = 128;
  vec bandwidths = linspace<vec>(0.1,10,BW);
  mat data = mat(BW,3);
  for(uint i = 0; i < BW; i++){
    cout << "Trial " << i << "..." << endl;
    vec rand_gaussian = gaussian(points, zeros<vec>(2), bandwidths(i));
    sp_mat extended_value_basis = sp_mat(orth(join_horiz(mat(value_basis),
                                                         rand_gaussian)));
    PLCP plcp = approx_lcp(extended_value_basis,smoother,
                           blocks,Q,free_vars);
    SolverResult sol = psolver.aug_solve(plcp);
    mat P = reshape(sol.p,N,A+1);
    vec V = P.col(0);
    vec res = bellman_residual_at_nodes(&mesh,&di,V,GAMMA);
    vec res_norm = vec{norm(res,1),norm(res,2),norm(res,"inf")};

    data.row(i) = (res_norm - ref_res_norm).t();
    cout << data.row(i);
  }
  
  mesh.write_cgal("test.mesh");
  Archiver arch = Archiver();
  arch.add_vec("ref_res",ref_res);
  arch.add_mat("data",data);
  arch.add_mat("bandwidths",bandwidths);

  arch.write("test.data");
}
