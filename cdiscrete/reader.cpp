#include <assert.h>
#include <iostream>
#include <armadillo>

#include "reader.h"


using namespace arma;

void read_mcts_config_file(Demarshaller & demarsh,
			   RegGrid & grid,
			   Problem & problem,
			   MCTSContext & context,
			   uint & sim_horizon,
			   mat & start_states,
			   RegGrid & ref_grid,
			   vec & ref_v){

  // Read in grid
  grid.low = demarsh.get_vec("Low boundary");
  grid.high = demarsh.get_vec("High boundary");
  grid.num_cells = conv_to<uvec>::from(demarsh.get_vec("Number of cells"));
  assert(check_dim(grid,2)); // 2D problem
  assert(verify(grid));

  
  // Physics params
  double step_len =  demarsh.get_scalar("Step length");
  uint n_steps = (uint) demarsh.get_scalar("Number of steps");
  double damp = demarsh.get_scalar("Dampening");
  double jitter = demarsh.get_scalar("Control jitter");

  // Other aspects of the problem
  double cost_radius = demarsh.get_scalar("Cost ball radius");
  double discount = demarsh.get_scalar("Discount");
  mat actions = demarsh.get_mat("Actions");

  // Assemble problem
  DoubleIntegrator * di_fn = new DoubleIntegrator(step_len,
					    n_steps,
					    damp,
					    jitter);
  BoundaryEnforcer * bnd_di_fn = new BoundaryEnforcer(di_fn,grid);
  BallCost * cost_fn = new BallCost(cost_radius,zeros<vec>(2));
  problem.trans_fn = bnd_di_fn;
  problem.cost_fn  = cost_fn;
  problem.discount = discount;
  problem.actions  = actions;

  vec v = demarsh.get_vec("MCTS Value");
  mat flow = demarsh.get_mat("MCTS Flow");

  uint mcts_budget = (uint)demarsh.get_scalar("MCTS growth budget");
  double p_scale = demarsh.get_scalar("P term scale");
  double ucb_scale = demarsh.get_scalar("UCB term scale");
  uint rollout_horizon = (uint) demarsh.get_scalar("Rollout horizon");
  double q_min_step = demarsh.get_scalar("Q min update stepsize");
  uint update_ret_mode = (uint) demarsh.get_scalar("Update mode");
  uint action_select_mode = (uint)demarsh.get_scalar("Action select mode");

  // Assemble objects for MCTS context
  InterpFunction * v_fn = new InterpFunction(v,grid);

  // flow = [flow]_+
  flow(find(flow < 0)).fill(0);
  ProbFunction * prob_fn = new ProbFunction(flow,grid);
  //DIBangBangPolicy * rollout = new DIBangBangPolicy(problem.actions);
  DILinearPolicy * rollout = new DILinearPolicy(problem.actions);

  // Build MCTS context
  context.problem_ptr = & problem;
  context.n_actions = num_actions(problem.actions);
  context.v_fn = v_fn;
  context.prob_fn = prob_fn;
  context.rollout = rollout;
  context.p_scale = p_scale;
  context.ucb_scale = ucb_scale;
  context.rollout_horizon = rollout_horizon;
  context.q_min_step = q_min_step;
  context.update_ret_mode = update_ret_mode;
  context.action_select_mode = action_select_mode;
  context.mcts_budget = mcts_budget;

  // Simulation stuff
  sim_horizon = (uint)demarsh.get_scalar("Simulation horizon");
  start_states = demarsh.get_mat("Start states");

  // Reference value function (for trunacting simulation runs)
  ref_grid.low = demarsh.get_vec("Low boundary");
  ref_grid.high = demarsh.get_vec("High boundary");
  ref_grid.num_cells = conv_to<uvec>::from(demarsh.get_vec("Number of cells"));
  ref_v =  demarsh.get_vec("Reference Value");

  assert(check_dim(ref_grid,2)); // 2D problem
  assert(verify(ref_grid));
    
}
