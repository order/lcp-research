#include <assert.h>
#include <iostream>

#include "costs.h"
#include "function.h"
#include "marshaller.h"
#include "mcts.h"
#include "policy.h"
#include "transfer.h"

#define EARLY_TERM_THRESH 0.01

using namespace std;

void read_grid(Demarshaller & demarsh,
	      RegGrid & grid){
  grid.low = demarsh.get_vec("Low boundary");
  grid.high = demarsh.get_vec("High boundary");
  grid.num_cells = conv_to<uvec>::from(demarsh.get_vec("Number of cells"));
  assert(check_dim(grid,2)); // 2D problem
  assert(verify(grid));

}

void read_problem(Demarshaller & demarsh,
		       RegGrid & grid,
		       Problem & problem){
  // Physics params
  double step_len =  demarsh.get_scalar("Step length");
  uint n_steps = (uint) demarsh.get_scalar("Number of steps");
  double damp = demarsh.get_scalar("Dampening");
  double jitter = demarsh.get_scalar("Control jitter");
  
  // Other aspects of the problem
  double cost_radius = demarsh.get_scalar("Cost ball radius");
  double discount = demarsh.get_scalar("Discount");
  mat actions = demarsh.get_mat("Actions");
  
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
}

void read_mcts_context(Demarshaller & demarsh,
			 RegGrid & grid,
			 Problem & problem,
			 MCTSContext & context){

  vec v = demarsh.get_vec("Value");
  mat q = demarsh.get_mat("Q");
  mat flow = demarsh.get_mat("Flow");

  double p_scale = demarsh.get_scalar("P term scale");
  double ucb_scale = demarsh.get_scalar("UCB term scale");
  uint rollout_horizon = (uint) demarsh.get_scalar("Rollout horizon");
  
  // Q estimates
  InterpFunction * v_fn = new InterpFunction(v,grid);
  InterpMultiFunction * q_fn = new InterpMultiFunction(q,grid);
    
  // INITIAL PROB
  InterpMultiFunction * flow_fn = new InterpMultiFunction(flow,grid);
  ProbFunction * prob_fn = new ProbFunction(flow_fn);
  
  // ROLLOUT
  DIBangBangPolicy * rollout = new DIBangBangPolicy(problem.actions);
  
  context.problem_ptr = & problem;
  context.n_actions = num_actions(problem.actions);

  context.v_fn = v_fn;
  context.q_fn = q_fn;
  context.prob_fn = prob_fn;
  context.rollout = rollout;

  context.p_scale = p_scale;
  context.ucb_scale = ucb_scale;
  context.rollout_horizon = rollout_horizon;
}

int main(int argc, char ** argv){

  if(2 != argc){
    std::cerr << "Usage: driver <mcts input file>" << std::endl;
    return -1;
  }
  
  string filename = string(argv[1]);
  std::cout << "Loading " << filename << std::endl;

  Demarshaller demarsh = Demarshaller(filename);
  assert( 19 == demarsh.get_num_objs());

  RegGrid grid;
  read_grid(demarsh,grid);
  
  Problem problem;
  read_problem(demarsh,grid,problem);
  
  MCTSContext context;
  read_mcts_context(demarsh,grid,problem,context);

  uint mcts_budget = (uint)demarsh.get_scalar("MCTS growth budget");

  uint sim_horizon = (uint)demarsh.get_scalar("Simulation horizon");
  mat start_states = demarsh.get_mat("Start states");
  

  uint N = start_states.n_rows;
  TransferFunction * t_fn = problem.trans_fn;
  mat actions = problem.actions;
  
  for(uint i = 0; i < N; i++){
    //for(uint i =0; i<1; i++){
    // Pick state
    std::cout << i << '/' << N << std::endl;
    //vec curr_state = ones<vec>(2);
    vec curr_state = start_states.row(i).t();

    for(uint t = 0; t < sim_horizon; t++){
    //for(uint t = 0; t < 5; t++){

      //Build tree
      MCTSNode * root = new MCTSNode(curr_state, &context);
      add_root(&context,root);
      grow_tree(root,mcts_budget);
      //root->print_debug();

      // Move to next action
      vec action = actions.row(root->get_best_action()).t();
      curr_state = t_fn->get_next_state(curr_state,
					action);
      delete_tree(&context);
      //std::cout<< "\t" << t << ": " << curr_state.t() << std::endl;
      if(norm(curr_state) < EARLY_TERM_THRESH){
	break;
      }
    }
  }
  delete_context(&context);
  return 0;
}
