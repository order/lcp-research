#include <assert.h>
#include <iostream>

#include "costs.h"
#include "function.h"
#include "marshaller.h"
#include "mcts.h"
#include "policy.h"
#include "transfer.h"

using namespace std;

int main(int argc, char ** argv){

  if(argc < 2 or argc > 3){
    std::cerr << "Usage: driver <mcts input file>" << std::endl;
    return -1;
  }
  string filename = string(argv[1]);
  bool save_sim = false;
  if(argc ==3 and 1 == atoi(argv[2])){
    save_sim = true;
  } 
  arma_rng::set_seed_random();

  // Read in config file
  Demarshaller demarsh = Demarshaller(filename);
  RegGrid grid;  
  Problem problem;  
  MCTSContext context;
  mat start_states;
  uint sim_horizon;
  vec ref_v;
  read_mcts_config_file(demarsh,
			grid,
			problem,
			context,
			sim_horizon,
			start_states
			ref_v);
  InterpFunction ref_v_fn = InterpFunction(ref_v,grid);

  
  uint N = start_states.n_rows;
  uint D = start_states.n_cols;
  assert(D == 2);
  
  TransferFunction * t_fn = problem.trans_fn;
  mat actions = problem.actions;
  cube traj;
  cube decisions;
  mat costs;
  if(save_sim){
    traj = cube(N,2,sim_horizon).fill(0);
    decisions = cube(N,1,sim_horizon).fill(0);
    costs = mat(N,sim_horizon).fill(0);
  }
  
  vec gains = zeros<vec>(N);
  for(uint i = 0; i < N; i++){
    // Pick state
    vec curr_state = start_states.row(i).t();
    vec last_state = 10*ones<vec>(2);
    uint t;
    for(t = 0; t < sim_horizon; t++){
      //Build tree
      MCTSNode * root = new MCTSNode(curr_state, &context);
      add_root(&context,root);
      grow_tree(root,context.mcts_budget);
      //root->print_debug();

      //Get action
      vec action;
      action = actions.row(root->get_action(context.action_select_mode)).t();

      // Record the cost and gain
      double cost = problem.cost_fn->get_cost(curr_state,action);
      
      gains(i) += pow(problem.discount,t) * cost;

      if(save_sim){
	traj.slice(t).row(i) = curr_state.t();
	decisions.slice(t).row(i) = action.t();
	costs(i,t) = cost;
      }

      // Transition
      last_state = curr_state;
      curr_state = t_fn->get_next_state(curr_state,
					action);
      if(norm(curr_state) < SIMTHRESH
	 and norm(last_state) < SIMTHRESH){
	break;
      }

      // Scrap tree.
      delete_tree(&context);
    }
    
    if(t == sim_horizon){
      // Add tail cost if didn't terminate early.
      double tail = pow(problem.discount,sim_horizon)
	* ref_v_fn->f(curr_state);
      gains(i) += tail;
    }
  }
  /*
  std::cout << "Mean: " << mean(gains) << std::endl;
  std::cout << "Median: " << median(gains) << std::endl;
  std::cout << "S.D.: " << stddev(gains) << std::endl;
  */
  std::cout << gains << std::endl;

  delete_context(&context);
  if(save_sim){
    Marshaller marsh;
    marsh.add_vec(gains);
    marsh.add_cube(traj);
    marsh.add_cube(decisions);
    marsh.add_mat(costs);
    marsh.save(filename + ".sim");
  }
  return 0;
}
