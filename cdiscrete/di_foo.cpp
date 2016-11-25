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
#define LENGTH 0.5
#define GAMMA 0.99
#define SMOOTH_BW 5
#define SMOOTH_THRESH 1e-4

#define RBF_GRID_SIZE 4
#define RBF_BW 0.25

vec new_vector(const Points & points){
  double angle = 1.2567;
  double b = 0.001;
  double c = 0.0785;
  mat rot = mat{{cos(angle), -sin(angle)},{sin(angle),cos(angle)}};
  mat cov = rot * diagmat(vec{b,c}) * rot.t();
  return gaussian(points,zeros<vec>(2),cov);
}

sp_mat make_value_basis(const Points & points){

  uint N = points.n_rows;
  
  uint k = RBF_GRID_SIZE;
  vec grid = linspace<vec>(-B,B,k);
  vector<vec> grids;
  grids.push_back(grid);
  grids.push_back(grid);

  mat centers = make_points(grids);
  double bandwidth = RBF_BW;
  mat basis = make_rbf_basis(points,centers,bandwidth,1e-6);
  vec new_vec = new_vector(points);
  

  basis = join_horiz(basis,new_vec);
  basis = orth(basis);
  //sp_mat basis = make_voronoi_basis(points,centers);
  //sp_mat basis = speye(N,N);
  return sp_mat(basis);
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

  // Shapes
  vec ball = zeros<vec>(N);
  vec dist = sqrt(sum(pow(points,2),1));
  ball(find(dist < 1)).fill(1);

  vec lball = spsolve(blocks.at(1).t(), ball);
  vec pball = spsolve(p_blocks.at(1).t(), ball);
  mesh.write_cgal("balls.mesh");
  Archiver barch = Archiver();
  barch.add_vec("lball",lball);
  barch.add_vec("pball",pball);
  barch.write("balls.data");
  exit(1);
  
  // Build smoother
  cout << "Building smoother matrix..." << endl;
  double bandwidth = SMOOTH_BW;
  double thresh = SMOOTH_THRESH;
  sp_mat smoother = gaussian_smoother(points,bandwidth,thresh);
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
  sp_mat value_basis = make_value_basis(points);
  PLCP plcp = approx_lcp(value_basis,smoother,blocks,Q,free_vars);

  //KojimaSolver solver = KojimaSolver();
  ProjectiveSolver solver = ProjectiveSolver();
  solver.comp_thresh = 1e-22;
  solver.initial_sigma = 0.25;
  solver.aug_rel_scale = 0.75;
  solver.regularizer = 0;
  solver.verbose = true;

  //SolverResult sol = solver.aug_solve(lcp);
  SolverResult sol = solver.aug_solve(plcp);
  
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
  uvec p_agg =  policy_agg(&mesh,&di,V,F,GAMMA);

  cout << "res_norm: " << res_norm.t();
  
  mesh.write_cgal("test.mesh");
  Archiver arch = Archiver();
  arch.add_mat("P",P);
  arch.add_mat("D",D);

  arch.add_vec("res",res);
  arch.add_vec("res_faces",res_faces);

  arch.add_vec("adv",adv);
  arch.add_uvec("p_agg",p_agg);

  arch.add_vec("new_vec",new_vector(points));
  
  arch.write("test.data");
}
