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
#define GAMMA 0.995
#define SMOOTH_BW 5
#define SMOOTH_THRESH 1e-4

#define RBF_GRID_SIZE 4
#define RBF_BW 0.25

#define NUM_BW 25

vec new_vector(const Points & points){
  double angle = 2.67;
  double b = 0.006;
  double c = 0.16;
  mat rot = mat{{cos(angle), -sin(angle)},{sin(angle),cos(angle)}};
  mat cov = rot * diagmat(vec{b,c}) * rot.t();
  return gaussian(points,zeros<vec>(2),cov);
}

sp_mat make_value_basis(const Points & points){
  uint N = points.n_rows;
  
  vec grid = linspace<vec>(-B,B,RBF_GRID_SIZE);
  vector<vec> grids;
  grids.push_back(grid);
  grids.push_back(grid);

  mat centers = make_points(grids);
  mat basis = make_rbf_basis(points,centers,RBF_BW,1e-6);
  vec new_vec = new_vector(points);
  basis = join_horiz(basis,new_vec);

  basis = orth(basis);
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
  psolver.comp_thresh = 1e-22;
  psolver.initial_sigma = 0.25;
  psolver.verbose = false;
  psolver.iter_verbose = false;

  cout << "Starting reference augmented PLCP solve..."  << endl;
  SolverResult ref_sol = psolver.aug_solve(ref_plcp);
  mat ref_P = reshape(ref_sol.p,N,A+1);
  mat ref_D = reshape(ref_sol.d,N,A+1);
  vec ref_V = ref_P.col(0);
  mat ref_F = ref_P.tail_cols(A);
  vec ref_res = bellman_residual_at_nodes(&mesh,&di,ref_V,GAMMA);
  
  vec ref_res_norm = vec{norm(ref_res,1),
			 norm(ref_res,2),
			 norm(ref_res,"inf"),
			 abs(accu(ref_res))};
  vec adv =  advantage_function(&mesh,&di,ref_V,GAMMA);
  uvec p_agg =  policy_agg(&mesh,&di,ref_V,ref_F,GAMMA);

  
  vec bandwidths = logspace<vec>(-2,0.75,NUM_BW); // 0.1 to ~6
  cube data = cube(N,NUM_BW,4); // l1,l2,linf,accu
  for(uint i = 0; i < N; i++){
    cout << "Center " << i << " of " << N << "..." << endl;
    for(uint j = 0; j < NUM_BW;j++){
      cout << "\tBandwidth " << bandwidths(j) << "..." << endl;
      vec rand_gaussian = gaussian(points,
				   points.row(i).t(), // location
				   bandwidths(j)); // Width
      sp_mat extended_value_basis = sp_mat(orth(join_horiz(mat(value_basis),
							   rand_gaussian)));
      PLCP plcp = approx_lcp(extended_value_basis,smoother,
			     blocks,Q,free_vars);
      SolverResult sol = psolver.aug_solve(plcp);
      mat P = reshape(sol.p,N,A+1);
      vec V = P.col(0);
      vec res = bellman_residual_at_nodes(&mesh,&di,V,GAMMA);
      vec res_norm = vec{norm(res,1),norm(res,2),
			 norm(res,"inf"),abs(accu(res))};

      data.tube(i,j) = (res_norm - ref_res_norm).t();
    }
  }
  
  mesh.write_cgal("test.mesh");
  Archiver arch = Archiver();
  arch.add_mat("P",ref_P);
  arch.add_mat("D",ref_D);
  arch.add_vec("ref_res",ref_res);
  arch.add_vec("adv",adv);
  arch.add_uvec("p_agg",p_agg);

  arch.add_mat("points",points);
  arch.add_vec("bandwidths",bandwidths);
  arch.add_cube("data",data);

  arch.write("test.data");
}
