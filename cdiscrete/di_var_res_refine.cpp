#include "di.h"
#include "grid.h"
#include "misc.h"
#include "io.h"
#include "lcp.h"
#include "solver.h"
#include "basis.h"
#include "smooth.h"
#include "refine.h"

/*
 * Double Integrator simulation with the variable resolution basis.
 *
 * This script tests out how Munos & Moore (Variable Resolution 
 * Discretization in Optimal Control; 1999) style variable resolution grid
 * refinement works for the Double Integrator 2D dynamics.
 *
 * Uses an underlying regular state-space discretization with a variable 
 * resolution "active" grid.
 */

// Bounding box [-B, B] x [-B, B]

// Power of 2 + 1
#define N_GRID_POINTS 65
#define N_INITIAL_SPLIT 3
#define N_OOB 1
#define UVEC_GRID_SIZE uvec{N_GRID_POINTS, N_GRID_POINTS}

#define SIM_ACTIONS vec{-1,1}
#define SIM_STEP 0.025
#define SIM_NOISE_STD 0.0

#define BOUNDARY 5.0
#define GAMMA 0.997

#define COMP_THRESH 1e-6

#define IGNORE_Q true

#define N_ADD_ROUNDS 6

mat build_bbox(){
  // Build the D x 2 bounding box
  mat bbox = mat(2,2);
  bbox.col(0).fill(-BOUNDARY);
  bbox.col(1).fill(BOUNDARY);
  return bbox;
}

MultiLinearVarResBasis make_basis(){
  /*
   * Set up the initial basis.
   * Uniformly splits each cell in each dimension N_INITIAL_SPLIT times.
   */
  uvec grid_size = UVEC_GRID_SIZE;
  MultiLinearVarResBasis basis_factory = MultiLinearVarResBasis(grid_size);
  basis_factory.split_per_dimension(0, uvec{N_INITIAL_SPLIT,N_INITIAL_SPLIT});
  cout << "Number of cells: " << basis_factory.m_cell_to_bbox.size() << endl;
  
  return basis_factory;
}

DoubleIntegratorSimulator build_di_simulator(){
  /*
   * Set up the continuous space simulator
   */
  mat bbox = build_bbox();
  return DoubleIntegratorSimulator(bbox, SIM_ACTIONS, SIM_NOISE_STD, SIM_STEP);
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
  psolver.comp_thresh = COMP_THRESH;
  psolver.initial_sigma = 0.25;
  psolver.verbose = true;
  psolver.iter_verbose = true;
  psolver.regularizer = 0;

  // Build the LCP
  PLCP plcp = approx_lcp(sp_basis,
			 smoother,
                         blocks,
			 Q,
			 free_vars,
			 IGNORE_Q);

  // Run the augmented solve
  Archiver arch = Archiver();
  arch.add_sp_mat("basis", plcp.P);
  arch.write("/home/epz/data/di_var_res_refine_basis.data");

  return psolver.aug_solve(plcp);
}

////////////////////////////////////////////////////////////////////////////
// Iteration result recording struct //
///////////////////////////////////////

class DiResultRecorder{
public:
  DiResultRecorder(uint n_nodes, uint n_actions, uint n_rounds);
  void record(const SolverResult & result);
  void write_to_file(const string & filename) const;

  vec get_vector(uint action, uint rount) const;
  
  vec get_last_value() const;
  vec get_value(uint round) const;

  vec get_agg_flow(uint round) const;

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

void DiResultRecorder::record(const SolverResult & result){
  assert(_curr_round < _n_rounds);
  mat P = reshape(result.p, _n_nodes, _n_actions+1);
  mat D = reshape(result.d, _n_nodes, _n_actions+1);
  _primals.slice(_curr_round) = P;
  _duals.slice(_curr_round) = D;
  _curr_round++;
}

void DiResultRecorder::write_to_file(const string & filename) const{
  Archiver arch = Archiver();
  arch.add_cube("primals", _primals);
  arch.add_cube("duals", _duals);
  arch.write(filename);
}

vec DiResultRecorder::get_vector(uint action, uint round) const{
  assert(action <= _n_actions);
  assert(round < _n_rounds);

  return _primals.subcube(
			  0, action, round,
			  _n_nodes - 1, action, round);
}

vec DiResultRecorder::get_agg_flow(uint round) const{
  assert(round < _n_rounds);

  vec agg_flow = sum(_primals.subcube(
				      0, 1, round,
				      _n_nodes - 1, _n_actions, round
				      ), 1);
  assert(_n_nodes == agg_flow.n_elem);
  return agg_flow;
}

vec DiResultRecorder::get_value(uint round) const{
  return get_vector(0, round);
}


vec DiResultRecorder::get_last_value() const{
  // Get the final value vector.
  return get_value(_n_rounds - 1);
}


////////////////////////////////////////////////////////////////////////////
// MAIN FUNCTION //
///////////////////

int main(int argc, char** argv)
{
  // Set up 2D space
  cout << "Generating underlying discretization..." << endl;
  mat bbox = build_bbox();
  uvec grid_size = UVEC_GRID_SIZE - 1;  // n_grid -> n_cells
  UniformGrid grid = UniformGrid(bbox, grid_size, N_OOB);

  TypedPoints points = grid.get_all_nodes();
  uint n_points = points.n_rows;
  assert(n_points > 0);

  // Set up the simulator
  DoubleIntegratorSimulator di = build_di_simulator();
  uint n_actions = di.num_actions();
  assert(n_actions >= 2);

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
  MultiLinearVarResBasis basis_factory = make_basis();

  // Set up per-iteration result data structures
  DiResultRecorder recorder = DiResultRecorder(n_points,
					       n_actions,
					       N_ADD_ROUNDS);

  // Run the basis refinement rounds
  for(uint I = 0; I < N_ADD_ROUNDS; I++){
    cout << "Running basis improvement round: "
	 << (I + 1) << "/" << N_ADD_ROUNDS  << endl;

    cout << "Generating basis..." << endl;
    sp_mat sp_basis = basis_factory.get_basis();
    sp_basis = sp_mat(orth(mat(sp_basis))); // Yeah, I know.
    
    cout << "Starting PLCP solve..." << endl;
    SolverResult sol = find_solution(sp_basis, blocks, Q, free_vars);
    recorder.record(sol);

    if(I < N_ADD_ROUNDS - 1){
      // Compute the splitting heuristic
      vec value = recorder.get_value(I);
      vec agg_flow = recorder.get_agg_flow(I);
      vec cell_agg_flow = basis_factory.get_cell_mean_reduced(agg_flow);
      vec cell_vars = basis_factory.get_cell_var_reduced(value);
      vec heur = cell_vars % cell_agg_flow;  // NB: OOB not a cell

      // Right now just do the single max
      uint split_id = cell_vars.index_max();
      cout << "Splitting: " << split_id << endl;
      cout << "Variance: " << cell_vars(split_id) << endl;
      basis_factory.split_per_dimension(split_id, uvec{1,1});
    } 
  }
  vec value = recorder.get_last_value();
  vec cell_vars = basis_factory.get_cell_var(value);
  Archiver arch = Archiver();
  arch.add_vec("x", value);
  arch.add_vec("y", cell_vars);
  arch.write("/tmp/cell_var.arch");
  
  // Write to file
  recorder.write_to_file("/home/epz/data/di_var_res_refine.data");
}
