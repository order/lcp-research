#include <iostream>
#include "costs.h"
#include "function.h"
#include "mcts.h"
#include "policy.h"
#include "transfer.h"

using namespace std;

int main(int argc, char ** argv){

  double gamma = 0.99;
  double dilate = 1.0 / (1.0 - gamma);

  // READ IN FROM PYTHON
  vec v =       dilate*ones<vec>(442);
  mat q =       dilate*ones<mat>(442,3);
  mat flow =    ones<mat>(442,3);
  mat actions = mat("-1;0;1");


  RegGrid grid;
  grid.low = -5*ones<vec>(2);
  grid.high = 5*ones<vec>(2);
  grid.num_cells = 20*ones<uvec>(2);

  
  // Q estimates
  InterpMultiFunction q_fn = InterpMultiFunction(q,grid);
  InterpFunction v_fn = InterpFunction(v,grid);
    
  // INITIAL PROB
  InterpMultiFunction flow_fn = InterpMultiFunction(flow,grid);
  ProbFunction prob_fn = ProbFunction(&flow_fn);

  // ROLLOUT
  DIBangBangPolicy rollout = DIBangBangPolicy(actions);

  // REST OF CONTEXT
  DoubleIntegrator di_fn = DoubleIntegrator(0.01,5,1e-5,0.1); // TODO: pass in
  BoundaryEnforcer bnd_di_fn = BoundaryEnforcer(&di_fn,grid);
  BallCost cost_fn = BallCost(0.15,zeros<vec>(2));


  Problem problem;
  problem.trans_fn = & bnd_di_fn;
  problem.cost_fn  = & cost_fn;
  problem.discount = gamma;
  problem.actions  = actions;
  
  MCTSContext context;
  context.problem_ptr = & problem;
  context.n_actions = 3;

  context.v_fn = &v_fn;
  context.q_fn = &q_fn;
  context.prob_fn = &prob_fn;
  context.rollout = &rollout;

  context.horizon = 100;
  context.p_scale = 1;
  context.ucb_scale = 5;

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
}
