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
#define SMOOTH_BW 5
#define SMOOTH_THRESH 1e-3

#define RBF_GRID_SIZE 1 // Make sure even so origin isn't already covered
#define RBF_BW 0.25

#define NUM_ADD_ROUNDS 8

#define NUM_ANGLE 8
#define NUM_AMPL  7

mat build_bbox(){
  mat bbox = mat(2,2);
  bbox.col(0).fill(-B);
  bbox.col(1).fill(B);
  return bbox;
}

mat make_basis(const Points & points){
  uint N = points.n_rows;
  
  vec grid = linspace<vec>(-B,B,RBF_GRID_SIZE);
  vector<vec> grids;
  grids.push_back(grid);
  grids.push_back(grid);

  mat centers = make_points(grids);
  mat basis = make_rbf_basis(points,centers,RBF_BW,1e-6);

  basis = orth(basis);
  return basis;
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

mat find_primal(const mat & basis,
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
  
  PLCP plcp = approx_lcp(sp_mat(basis),smoother,
                         blocks,Q,free_vars);
  SolverResult sol = psolver.aug_solve(plcp);
  mat P = reshape(sol.p,N,A+1);
  return P;
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
  mat heuristics = mat(N,NUM_ADD_ROUNDS+1);
  cube primals = cube(N,A+1,NUM_ADD_ROUNDS+1);
  mat centers = mat(NUM_ADD_ROUNDS,2);
  mat params = mat(NUM_ADD_ROUNDS,3);

  for(uint I = 0; I < NUM_ADD_ROUNDS; I++){
    cout << "Running " << I << "/" << NUM_ADD_ROUNDS  << endl;

    // Pick the location

    mat P = find_primal(basis,smoother,blocks,Q,free_vars);
    vec res = find_residual(mesh,di,P);
    vec heur = heuristic(P,res);
    residuals.col(I) = res;
    primals.slice(I) = P;
    heuristics.col(I) = heur;

    uint idx = sample_vec(heur);
    vec center = points.row(idx).t();
    cout << "Center:" << center.t();
    cout.flush();
    vec angles = linspace(0,datum::pi,NUM_ANGLE+1).head(NUM_ANGLE);
    vec amplitudes = logspace<vec>(-1.5,0.5,NUM_AMPL); // 0.01 to 1

    double best_norm = datum::inf;
    vec best_param;
    mat best_basis;
    
    for(uint a = 0; a < NUM_ANGLE; a++){
      cout << "\tAngle: " << angles(a) << endl;
      for(uint i = 0; i < NUM_AMPL; i++){
        for(uint j = i; j < NUM_AMPL; j++){
          vec test_param = vec{angles(a),
                           amplitudes(i),
                           amplitudes(j)};
          mat test_basis = mat(N,2 + basis.n_cols);
          test_basis.col(0) = new_vector(points,center,test_param);
          test_basis.col(1) = new_vector(points,-center,test_param);
          test_basis.tail_cols(basis.n_cols) = basis;
          test_basis = orth(test_basis);

          mat test_P = find_primal(test_basis,smoother,blocks,Q,free_vars);
          vec test_res = find_residual(mesh,di,test_P);
          vec test_heur = heuristic(test_P,test_res);

          double test_norm = norm(test_heur,1);
          if(test_norm < best_norm){
            cout << "\t\tNew best: " << test_param.t();
            cout << "\t\tHeuristic 1-norm: " << test_norm << endl;;
            
            best_norm = test_norm;
            best_param = test_param;
            best_basis = test_basis;
          }
        }
      }
    }
        
    // Record best params
    centers.row(I) = center.t();
    params.row(I) = best_param.t();

    //Extend vector
    basis = best_basis;
  }
  mat P = find_primal(basis,smoother,blocks,Q,free_vars);
  vec res = find_residual(mesh,di,P);
  vec heur = heuristic(P,res);
  primals.slice(NUM_ADD_ROUNDS) = P;
  residuals.col(NUM_ADD_ROUNDS) = res;
  heuristics.col(NUM_ADD_ROUNDS) = heur;

  mesh.write_cgal("test.mesh");
  Archiver arch = Archiver();
  arch.add_mat("residuals",residuals);
  arch.add_cube("primals",primals);
  arch.add_mat("centers",centers);
  arch.add_mat("heuristics",heuristics);
  arch.add_mat("params",params);
  arch.add_vec("new_vec",new_vector(points,
                                    centers.tail_rows(1).t(),
                                    params.tail_rows(1).t()));

  arch.write("test.data");
}
