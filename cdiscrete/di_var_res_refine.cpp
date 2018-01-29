#include "di.h"
#include "misc.h"
#include "io.h"
#include "lcp.h"
#include "solver.h"
#include "basis.h"
#include "smooth.h"
#include "refine.h"

/*
 * Double Integrator simulation with the variable resolution basis
 */

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace tri_mesh;

# define B 5.0

mat build_bbox(){
  mat bbox = mat(2,2);
  bbox.col(0).fill(-B);
  bbox.col(1).fill(B);
  return bbox;
}

mat make_basis(const TypedPoints & points){
  /*
   * Makes a variable resolution cell basis
   */
  uint N = points.n_rows;
  
}

DoubleIntegratorSimulator build_di_simulator(){
  /*
   * Set up the continuous space simulator
   */
  mat bbox = build_bbox();
  mat actions = vec{-1,1};
  double noise_std = 0.0;
  double step = 0.025;
  return DoubleIntegratorSimulator(bbox,actions,noise_std,step);
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


////////////////////
// MAIN FUNCTION //
///////////////////

int main(int argc, char** argv)
{
  // Set up 2D space
  cout << "Generating initial mesh..." << endl;
  mat bbox = build_bbox();
  UniformGrid grid = UniformGrid(bbox, uvec{N_GRID_POINTS, N_GRID_POINTS});

  TypedPoints points = grid.get_all_nodes();
  uint N = points.n_rows;
  assert(N > 0);
  
  DoubleIntegratorSimulator di = build_di_simulator();
  uint A = di.num_actions();
  assert(A >= 2);

  // Reference blocks
  cout << "Building LCP blocks..." << endl;
  vector<sp_mat> blocks = di.lcp_blocks(&grid, GAMMA);
  assert(A == blocks.size());
  assert(size(N,N) == size(blocks.at(0)));

  // Build the Q matrix
  cout << "Building RHS Q..." << endl;
  mat Q = di.q_mat(&grid);

  // Set up the free variables (rest are non-negative variables)
  bvec free_vars = zeros<bvec>((A+1)*N);
  free_vars.head(N).fill(1);

  cout << "Making value basis..." << endl;
  mat basis = make_basis(points);
  basis = orth(basis);

  cube primals = cube(N,A+1,NUM_ADD_ROUNDS+1);
  cube duals = cube(N,A+1,NUM_ADD_ROUNDS+1);
  
  mat residuals = mat(N,NUM_ADD_ROUNDS+1);
  mat min_duals = mat(N,NUM_ADD_ROUNDS+1);
  mat advantages = mat(N,NUM_ADD_ROUNDS+1);
  umat policies = umat(N,NUM_ADD_ROUNDS+1);

  mat misc = zeros(N,NUM_ADD_ROUNDS+1);
  
  for(uint I = 0; I < NUM_ADD_ROUNDS; I++){
    cout << "Running " << I << "/" << NUM_ADD_ROUNDS  << endl;
    // Pick the location
    SolverResult sol = find_solution(basis,smoother,blocks,Q,free_vars);
    mat P = reshape(sol.p,N,A+1);
    mat D = reshape(sol.d,N,A+1);
    vec V = P.col(0);
    mat F = P.tail_cols(A);
    
    vec res = find_residual(mesh,di,P);
    vec md_res = min(D.tail_cols(A),1);   
    uvec pi = index_max(F,1);
    vec adv = max(D.tail_cols(A),1) - md_res;
    
    mat new_vects = mat(N,2);
    uint idx = sample_vec(md_res);
    cout << "Index: " << idx << endl;

    for(uint i = 0; i < A; i++){
      //vec heur = md_res % gaussian(points,points.row(idx).t(),0.5);
      vec heur;
      if (norm(md_res,"inf") < 2){
	heur = res;
      }
      else{
	heur = md_res;
      }
      uvec mask = find((1-i) == pi);
      heur(mask).fill(0); // mask out
     
      new_vects.col(i) = spsolve(smoother * blocks.at(i).t(),heur);
      misc.col(I) += blocks.at(i) * heur;
    }


    basis = join_horiz(basis,new_vects);
    basis = orth(basis);
    
    primals.slice(I) = P;    
    duals.slice(I) = D;
    residuals.col(I) = res;
    min_duals.col(I) = md_res;
    advantages.col(I) = adv;
    policies.col(I) = pi;

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
  vec md_res = min(D.tail_cols(A),1);   
  uvec pi = index_max(P.tail_cols(A),1);
  vec adv = max(D.tail_cols(A),1) - md_res;

  primals.slice(NUM_ADD_ROUNDS) = P;    
  duals.slice(NUM_ADD_ROUNDS) = D;
  residuals.col(NUM_ADD_ROUNDS) = res;
  min_duals.col(NUM_ADD_ROUNDS) = md_res;
  advantages.col(NUM_ADD_ROUNDS) = adv;
  policies.col(NUM_ADD_ROUNDS) = pi;
		 misc.col(NUM_ADD_ROUNDS) = blocks.at(0) * res;


  mesh.write_cgal("test.mesh");
  Archiver arch = Archiver();
  arch.add_cube("primals",primals);
  arch.add_cube("duals",duals);
  arch.add_mat("residuals",residuals);
  arch.add_mat("min_duals",min_duals);
  arch.add_mat("advantages",advantages);
  arch.add_mat("misc",misc);

  arch.add_umat("policies",policies);

  arch.write("test.data");
}
