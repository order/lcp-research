#include <iostream>
#include "mcts.h"

int main(int argc, char ** argv){
  DoubleIntegrator di_fn = DoubleIntegrator(0.01,5,1e-5,0);
  BoundaryEnforcer bounded_di_fn = BoundaryEnforcer(&di_fn,boundary);
  
  MCTSContext context;
  context.trans_fn = & bounded_di_fn;
  context.cost_fn = & BallCosts(0.15,zeros<vec>(2));
  context.discount = 0.997;

  // Q function
  // Action prob
  // Policy
}
