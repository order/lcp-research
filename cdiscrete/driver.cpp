#include <iostream>
#include "costs.h"
#include "function.h"
#include "mcts.h"
#include "policy.h"
#include "transfer.h"

int main(int argc, char ** argv){
  DoubleIntegrator di_fn = DoubleIntegrator(0.01,5,1e-5,0);
 
  Boundary boundary;
  boundary.low = vec(2).fill(-5);
  boundary.high = vec(2).fill(5);  
  BoundaryEnforcer bounded_di_fn = BoundaryEnforcer(&di_fn,boundary);
  
  mat actions = mat("-1;0;1");
 
  MCTSContext context;
  BallCost cost_fn = BallCost(0.15,zeros<vec>(2));
  context.trans_fn = & bounded_di_fn;
  context.cost_fn = & cost_fn;
  context.discount = 0.997;

  ConstMultiFunction q_fn = ConstMultiFunction(3,1.0);
  ProbFunction prob_fn = ProbFunction(&q_fn);
  DIBangBangPolicy rollout = DIBangBangPolicy(actions);
  context.q_fn = &q_fn;
  context.prob_fn = &prob_fn;
  context.rollout = &rollout;

  context.actions = &actions;
  context.n_actions = 3;

  context.p_scale = 1;
  context.ucb_scale = 2;

  vec state = ones<vec>(2);
  MCTSNode root = MCTSNode(state, &context);
  root.add_child(0,randu<vec>(2));
  root.add_child(0,randu<vec>(2));  
  root.add_child(1,randu<vec>(2));  

  //node.print_debug();
  write_dot_file("test.dot", &root);
  print_nodes(context);
  
}
