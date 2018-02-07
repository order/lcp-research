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

MultiLinearVarResBasis make_basis(){

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

SolverResult find_solution(const sp_mat & sp_basis,
                const vector<sp_mat> & blocks,
                const mat & Q,
                const bvec & free_vars){
  /*
   * Set up and solve the projective LCP from basis and block information
   */
  uint n_points = sp_basis.n_rows;
  uint n_bases = sp_basis.n_cols;
  uint n_actions = blocks.size();
  assert(size(n_points, n_actions + 1) == size(Q));
  assert(size(n_points, n_points) == size(blocks.at(0)));

  // Trivial smoothing matrix
  sp_mat smoother = speye(n_points, n_points);
  
  // Set up the solver options
  ProjectiveSolver psolver = ProjectiveSolver();
  psolver.comp_thresh = 1e-12;
  psolver.initial_sigma = 0.25;
  psolver.verbose = true;
  psolver.iter_verbose = true;
  psolver.regularizer = 1e-12;

  // Build the LCP
  PLCP plcp = approx_lcp(basis,
			 smoother,
                         blocks,
			 Q,
			 free_vars);

  // Run the augmented solve
  return psolver.aug_solve(plcp);
}

////////////////////////////////////////////////////////////////////////////
// Iteration result recording struct //
///////////////////////////////////////

class DiResultRecorder{
public:
  DiResultRecorder(uint n_nodes, uint n_actions, uint n_rounds);
  void record(const SolutionResult & result);
  void write_to_file(const string & filename) const;

  cube _primals;
  cube _duals;
  uint _n_nodes;
  uint _n_actions;
  uint _n_rounds;
  uint _curr_round;
};

DiResultRecorder::DiResultRecorder(uint n_nodes,
				   uint n_actions,
				   uint n_rounds){
  _primals = cube(n_nodes, n_actions+1, n_rounds);
  _duals = cube(n_nodes, n_actions+1, n_rounds);
  _n_nodes = n_nodes;
  _n_actions= n_actions;
  _n_rounds = n_rounds;
  _curr_round = 0;
}

void DiResultRecorder::record(const SolutionResult & result){
  assert(_curr_round < _n_rounds);
  mat P = reshape(result.p, _n_nodes, _n_actions+1);
  mat D = reshape(result.d, _n_nodes, _n_actions+1);
  primals.slice(_curr_round) = P;    
  duals.slice(_curr_round) = D;
  _curr_round++;
}

void DiResultRecorder::write_to_file(const string & filename) const{
  Archiver arch = Archiver();
  arch.add_cube("primals", _primals);
  arch.add_cube("duals", _duals);
  arch.write(filename);
}


////////////////////////////////////////////////////////////////////////////
// MAIN FUNCTION //
///////////////////

int main(int argc, char** argv)
{
  // Set up 2D space
  cout << "Generating underlying discretization..." << endl;
  mat bbox = build_bbox();
  UniformGrid grid = UniformGrid(bbox, uvec{N_GRID_POINTS, N_GRID_POINTS});

  TypedPoints points = grid.get_all_nodes();
  uint n_points = points.n_rows;
  assert(N > 0);

  // Set up the simulator
  DoubleIntegratorSimulator di = build_di_simulator();
  uint n_actions = di.num_actions();
  assert(A >= 2);

  // Build the "exact" LCP P blocks
  cout << "Building LCP blocks..." << endl;
  vector<sp_mat> blocks = di.lcp_blocks(&grid, GAMMA);
  assert(n_actions == blocks.size());
  assert(size(n_points, n_points) == size(blocks.at(0)));

  // Build the Q matrix
  cout << "Building RHS Q..." << endl;
  mat Q = di.q_mat(&grid);

  // Set up the free variables (rest are non-negative variables)
  bvec free_vars = zeros<bvec>((n_actions+1) * n_points);
  free_vars.head(n_points).fill(1);

  // Set up the value basis factory
  cout << "Making value basis factory..." << endl;
  
  MultiLinearVarResBasis basis_factory = make_basis(points, bbox);

  // Set up per-iteration result data structures
  DiResultRecorder recorder = DiResultRecorder(n_points,
					       n_actions,
					       NUM_ADD_ROUNDS);
  
  
  for(uint I = 0; I < NUM_ADD_ROUNDS; I++){
    cout << "Running " << I << "/" << NUM_ADD_ROUNDS  << endl;

    cout << "Generating basis..." << endl;
    sp_mat sp_basis = basis_factor.get_basis();
    // TODO: ensure orthogonal
    
    cout << "Starting PLCP solve..." << endl;
    SolverResult sol = find_solution(basis, blocks, Q, free_vars);
    recorder.record(sol);

    // TODO: split the basis based on some Munos & Moore criterion
    
  }
  recorder.write_to_file("/home/epz/data/di_var_res_refine.data");

}
