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
#define LENGTH 0.35
#define GAMMA 0.995
#define SMOOTH_BW 1e9
#define SMOOTH_THRESH 1e-3

#define NUM_ADD_ROUNDS 16

mat build_bbox(){
  mat bbox = mat(2,2);
  bbox.col(0).fill(-B);
  bbox.col(1).fill(B);
  return bbox;
}

mat make_basis(const Points & points){
  uint N = points.n_rows;

  // General basis
  vector<vec> grids;
  grids.push_back(linspace<vec>(-B,B,4));
  grids.push_back(linspace<vec>(-B,B,4));
  Points grid_points = make_points(grids);
  
  mat basis = make_rbf_basis(points,grid_points,0.25,1e-5);
  basis = join_horiz(basis,gaussian(points,zeros<vec>(2),1));

  //mat basis = mat(N,3);
  //basis.col(0) = ones<mat>(N,1);
  //basis.col(1) = gaussian(points,zeros<vec>(2),0.25);
  //basis.col(2) = gaussian(points,zeros<vec>(2),1);
  return basis; // Don't normalize here
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

vec new_vector(const Points & points,
               const vec & center,
               const vec & param){
  assert(3 == param.n_elem);
  double a = param(0); // Angle
  double b = param(1); // Along angle length
  double c = param(2); // Orthogonal length
  
  mat rot = mat{{cos(a), -sin(a)},{sin(a),cos(a)}};
  mat cov = rot * diagmat(vec{b,c}) * rot.t();
  return gaussian(points,center,cov);
}

SolverResult find_solution(const mat & basis,
                const sp_mat & smoother,
                const vector<sp_mat> & blocks,
                const mat & Q,
                const bvec & free_vars){
  uint N = basis.n_rows;
  assert(size(N,N) == size(smoother));
  uint A = blocks.size();
  assert(size(N,A+1) == size(Q));

  ProjectiveSolver psolver = ProjectiveSolver();
  psolver.comp_thresh = 1e-12;
  psolver.initial_sigma = 0.25;
  psolver.verbose = false;
  psolver.iter_verbose = false;
  psolver.regularizer = 1e-12;
  
  PLCP plcp = approx_lcp(sp_mat(basis),smoother,
                         blocks,Q,free_vars);
  return psolver.aug_solve(plcp);
}

vec find_residual(const TriMesh & mesh,
                  const DoubleIntegratorSimulator & di,
                  const mat & P){
  vec V = P.col(0);
  return bellman_residual_at_nodes(&mesh,&di,V,GAMMA);
}

vec softmax(const vec & v, double temp){
  return normalise(exp(temp * abs(v)),1);
}
vec quantile_cut(const vec & v, double q){
  vec w = abs(v);
  w(find(w < quantile(w,q))).fill(0);
  w /= accu(w);
  return w;
}

uint sample_vec(const vec & v){
  vec w = quantile_cut(v,0.9);
  
  uint N = w.n_elem;
  double agg = 0;
  vec roll_v = randu<vec>(1);
  double roll = roll_v(0);
  for(uint i = 0; i < N; i++){
    agg += w(i);
    if(agg >= roll){
      return i;
    }
  }
  assert(false);
}

vec heuristic(const mat & P, const vec & res){
  uint A = P.n_cols - 1;
  vec agg = sum(P.tail_cols(A),1);
  return abs(res) % softmax(agg,0.25);
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
  //mat basis = ones<mat>(N,1) / (double) N;
  mat basis = make_basis(points);
  
  mat residuals = mat(N,NUM_ADD_ROUNDS+1);
  cube primals = cube(N,A+1,NUM_ADD_ROUNDS+1);
  cube duals = cube(N,A+1,NUM_ADD_ROUNDS+1);
  mat vresiduals = mat(N,NUM_ADD_ROUNDS+1);
  for(uint I = 0; I < NUM_ADD_ROUNDS; I++){
    cout << "Running " << I << "/" << NUM_ADD_ROUNDS  << endl;
    // Pick the location
    SolverResult sol = find_solution(basis,smoother,blocks,Q,free_vars);
    mat P = reshape(sol.p,N,A+1);
    mat D = reshape(sol.d,N,A+1);
    
    vec res = find_residual(mesh,di,P);
    vec md_res = min(D.tail_cols(A),1);
   
    assert(N == res.n_elem);
    uvec pi = index_max(P.tail_cols(A),1);
    //uvec pi = q_policy_at_nodes(&mesh,&di,P.col(0),GAMMA,25);
    assert(N == pi.n_elem);
    
    mat new_vects = mat(N,2);
    for(uint i = 0; i < A; i++){
      vec heur = vec(md_res); // Use the dual residual estimate, not the real deal
      uvec mask = find((1-i) == pi);
      heur(mask).fill(0); // mask out
     
      new_vects.col(i) = spsolve(blocks.at(i).t(),heur);
    }


    basis = join_horiz(basis,new_vects);
    basis = orth(basis);

    residuals.col(I) = res;
    primals.slice(I) = P;
    duals.slice(I) = D;
    vresiduals.col(I) = blocks.at(0) * P.col(0) + blocks.at(1) * P.col(1) - Q.col(0);


    vec bellman_res_norm = vec{norm(res,1),norm(res,2),norm(res,"inf")};
    vec md_res_norm = vec{norm(md_res,1),norm(md_res,2),norm(md_res,"inf")};

    cout << "\tBellman residual norm:" << bellman_res_norm.t() << endl;
    cout << "\tMin. dual residual norm:" << md_res_norm.t() << endl;

    cout << "\tBasis size: " << basis.n_cols << endl;
  }
  SolverResult sol = find_solution(basis,smoother,blocks,Q,free_vars);
  mat P = reshape(sol.p,N,A+1);
  mat D = reshape(sol.d,N,A+1);
  vec res = find_residual(mesh,di,P);
  
  residuals.col(NUM_ADD_ROUNDS) = res;
  primals.slice(NUM_ADD_ROUNDS) = P;
  duals.slice(NUM_ADD_ROUNDS) = D;  

  mesh.write_cgal("test.mesh");
  Archiver arch = Archiver();
  arch.add_mat("residuals",residuals);
  arch.add_cube("primals",primals);
  arch.add_cube("duals",duals);
  arch.add_mat("vresiduals",vresiduals);

  arch.write("test.data");
}
