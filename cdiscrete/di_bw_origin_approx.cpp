#include "di.h"
#include "misc.h"
#include "io.h"
#include "lcp.h"
#include "solver.h"
#include "basis.h"
#include "smooth.h"
#include "refine.h"

/*
  Program for exploring the effects of adding different gaussians
  to the origin for approximating a double integrator problem.
  Record l1,l2,linf Bellmen residual changes
*/

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace tri_mesh;

#define B 5.0
#define LENGTH 0.5
#define GAMMA 0.995
#define SMOOTH_BW 1e9
#define SMOOTH_THRESH 1e-4

#define RBF_GRID_SIZE 4 // Make sure even so origin isn't already covered
#define RBF_BW 0.25

#define NUM_ANGLE 20
#define NUM_AMPL  20

sp_mat make_value_basis(const Points & points){
  uint N = points.n_rows;
  
  vec grid = linspace<vec>(-B,B,RBF_GRID_SIZE);
  vector<vec> grids;
  grids.push_back(grid);
  grids.push_back(grid);

  mat centers = make_points(grids);
  mat basis = make_rbf_basis(points,centers,RBF_BW,1e-6);

  basis = orth(basis);
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

  //////////////////////////
  // SETUP FOR EXPERIMENT //
  //////////////////////////
  vec angles = linspace(0,datum::pi,NUM_ANGLE+1).head(NUM_ANGLE);
  vec amplitudes = logspace<vec>(-3,0,NUM_AMPL); // 0.01 to 1

  uint num_points = (NUM_ANGLE * (NUM_AMPL + 1) * NUM_AMPL) / 2;
  mat coords = mat(num_points,3);
  mat res_diff = mat(num_points,3);
  uint I = 0;
  
  for(uint a = 0; a < NUM_ANGLE; a++){
    double angle = angles(a);
    mat rot = mat{{cos(angle), -sin(angle)},{sin(angle),cos(angle)}};
    for(uint i = 0; i < NUM_AMPL; i++){
      double long_vec = amplitudes(i);
      for(uint j = i; j < NUM_AMPL; j++){
	double short_vec = amplitudes(j);
	vec coord = vec{angle,long_vec,short_vec};
	assert(I < num_points);
	coords.row(I) = coord.t();
	cout << "Running " << I << "/" << num_points  << endl;
	cout << "\tCoord: " << coord.t();

	mat cov = rot * diagmat(vec{long_vec,short_vec}) * rot.t();
	vec new_vec = gaussian(points,zeros<vec>(2),cov);
	sp_mat new_basis = sp_mat(orth(join_horiz(mat(value_basis),
						  new_vec)));

	PLCP plcp = approx_lcp(new_basis,smoother,
			       blocks,Q,free_vars);

	SolverResult sol = psolver.aug_solve(plcp);

	// Interpret results
	mat P = reshape(sol.p,N,A+1);
	vec V = P.col(0);
	vec res = bellman_residual_at_nodes(&mesh,&di,V,GAMMA);
	vec res_norm = vec{norm(res,1),norm(res,2),norm(res,"inf")};
	res_diff.row(I++) = (res_norm - ref_res_norm).t();
      }
    }
  }
  assert(I == num_points);
  
  mesh.write_cgal("test.mesh");
  Archiver arch = Archiver();
  arch.add_vec("ref_res",ref_res);
  arch.add_mat("res_diff",res_diff);
  arch.add_mat("coords",coords);

  arch.write("test.data");
}
