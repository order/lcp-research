#include <assert.h>
#include <iostream>

#include "costs.h"
#include "function.h"
#include "marshaller.h"
#include "mcts.h"
#include "policy.h"
#include "transfer.h"

using namespace std;

void set_up_di_problem(const RegGrid & grid,
		       double step_len,
		       double n_steps,
		       double damp,
		       double jitter,
		       double discount,
		       double cost_radius,
		       const mat & actions,
		       Problem & problem){
  DoubleIntegrator di_fn = DoubleIntegrator(step_len,
					    n_steps,
					    damp,
					    jitter);
  BoundaryEnforcer bnd_di_fn = BoundaryEnforcer(&di_fn,grid);
  BallCost cost_fn = BallCost(cost_radius,zeros<vec>(2));

  problem.trans_fn = & bnd_di_fn;
  problem.cost_fn  = & cost_fn;
  problem.discount = discount;
  problem.actions  = actions;
}

void set_up_mcts_context(const vec & v,
			 const mat & q,
			 const mat & flow,
			 const RegGrid & grid,
			 Problem & problem,
			 double p_scale,
			 double ucb_scale,
			 uint rollout_horizon,
			 MCTSContext & context){
  // Q estimates
  InterpMultiFunction q_fn = InterpMultiFunction(q,grid);
  InterpFunction v_fn = InterpFunction(v,grid);
    
  // INITIAL PROB
  InterpMultiFunction flow_fn = InterpMultiFunction(flow,grid);
  ProbFunction prob_fn = ProbFunction(&flow_fn);
  
  // ROLLOUT
  DIBangBangPolicy rollout = DIBangBangPolicy(problem.actions);
  
  context.problem_ptr = & problem;
  context.n_actions = num_actions(problem.actions);

  context.v_fn = &v_fn;
  context.q_fn = &q_fn;
  context.prob_fn = &prob_fn;
  context.rollout = &rollout;

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
  assert( 10 == demarsh.get_num_objs());

  // Physics params
  double step_len =  demarsh.get_scalar();
  uint n_steps = (uint) demarsh.get_scalar();
  double damp = demarsh.get_scalar();
  double jitter = demarsh.get_scalar();

  // Disc params
  RegGrid grid;
  grid.low = demarsh.get_vec();
  grid.high = demarsh.get_vec();
  grid.num_cells = conv_to<uvec>::from(demarsh.get_vec());

  // Other aspects of the problem
  double cost_radius = demarsh.get_scalar();
  double discount = demarsh.get_scalar();
  mat actions = demarsh.get_mat();

  /*
  vec v = demarsh.get_vec();
  mat q = demarsh.get_mat();
  mat flow = demarsh.get_mat();
  mat start_states = demarsh.get_mat();
  */


  Problem problem;
  set_up_di_problem(grid,
		    step_len,
		    n_steps,
		    damp,
		    jitter,
		    discount,
		    cost_radius,
		    actions,
		    problem);
    /*
  MCTSContext context;

  // Create root node
  vec root_state = vec("-1,1");
  for(uint i = 0; i < 1; i++){
    MCTSNode * root = new MCTSNode(root_state, &context);
    add_root(&context,root);
    
    grow_tree(root,2);
    //write_dot_file("test.dot",root);
    root->print_debug();
    delete_tree(&context);
  }
    */
}
