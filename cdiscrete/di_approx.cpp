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
#define SMOOTH_BW 100
#define RBF_GRID_SIZE 13

sp_mat make_value_basis(const Points & points){

  uint N = points.n_rows;
  
  uint k = RBF_GRID_SIZE;
  vec grid = linspace<vec>(-B,B,k);
  vector<vec> grids;
  grids.push_back(grid);
  grids.push_back(grid);

  mat centers = make_points(grids);
  double bandwidth = 2.5;
  mat basis = make_rbf_basis(points,centers,bandwidth,0);
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
  return DoubleIntegratorSimulator(bbox,actions);
}

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
  double bandwidth = SMOOTH_BW;
  double thresh = 1e-6;  
  sp_mat smoother = gaussian_smoother(points,bandwidth,thresh);
  assert(size(N,N) == size(smoother));

  // Build and pertrub the q
  cout << "Building RHS Q..." << endl;
  mat Q = di.q_mat(&mesh);
  
  bvec free_vars = zeros<bvec>((A+1)*N);
  free_vars.head(N).fill(1);

  cout << "Assembling blocks into smoothed projective LCP..." << endl;
  sp_mat value_basis = make_value_basis(points);

  vec rand_gaussian = gaussian(points, 4*randu<vec>(2) - 2, 3);
  sp_mat extended_value_basis = sp_mat(orth(join_horiz(mat(value_basis),
						       rand_gaussian)));
  PLCP plcp1 = approx_lcp(value_basis,smoother,blocks,Q,free_vars);
  PLCP plcp2 = approx_lcp(extended_value_basis,smoother,blocks,Q,free_vars);

  cout << "Initializing solver..." << endl;
  ProjectiveSolver solver = ProjectiveSolver();
  solver.comp_thresh = 1e-12;
  solver.initial_sigma = 0.25;
  solver.verbose = true;
    
  cout << "Starting augmented LCP solve..."  << endl;
  SolverResult sol1 = solver.aug_solve(plcp1);
  SolverResult sol2 = solver.aug_solve(plcp2);

  mat P1 = reshape(sol1.p,N,A+1);
  mat P2 = reshape(sol2.p,N,A+1);
  
  vec value1 = P1.col(0);
  vec value2 = P2.col(0);

  vec res1 =  bellman_residual_at_nodes(&mesh,&di,value1,GAMMA);  
  vec res2 =  bellman_residual_at_nodes(&mesh,&di,value2,GAMMA);  

  mesh.write_cgal("test.mesh");
  Archiver arch = Archiver();
  arch.add_vec("p",sol2.p);
  arch.add_vec("d",sol2.d);
  arch.add_vec("res1",res1);
  arch.add_vec("res2",res2);
  arch.add_vec("res_diff",res1 - res2);
  arch.write("test.sol");
}
