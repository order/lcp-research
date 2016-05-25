#include <iostream>
#include <assert.h>
#include <armadillo>

#include "misc.h"
#include "simulate.h"

using namespace arma;

void simulate(const mat & X0,
	      const Problem & problem,
	      const Policy & policy,
	      uint T,
	      SimulationOutcome & outcome){

  mat X = X0;
  uint N = X.n_rows;
  uint D = X.n_cols;
  uint A = policy.get_action_dim();
  
  outcome.points = cube(N,D,T);
  outcome.actions = cube(N,A,T);
  outcome.costs = mat(N,T);

  mat actions;
  vec costs;
  for(uint t = 0; t < T; t++){
    actions = policy.get_actions(X);
    costs = problem.cost_fn->get_costs(X,actions);

    
    outcome.points.slice(t) = X;
    outcome.actions.slice(t) = actions;
    outcome.costs.col(t) = costs;

    if(t < T-1){
      X = problem.trans_fn->get_next_states(X,actions);
    }
  }
}

void simulate_test(SimulationOutcome & res){
  /*
    Basic DI simulation with no arguments
   */
  uint T = 100;
  arma_rng::set_seed_random();
  mat x0 = 100*ones<mat>(1,2);
  std::cout << x0 << std::endl;
  
  mat actions = mat("-1;0;1");

  Boundary boundary;
  boundary.low=vec("-5,-5");
  boundary.high=vec("5,5");
  
  DoubleIntegrator di_fn = DoubleIntegrator(0.01,5,1e-5,0);
  BoundaryEnforcer bounded_di_fn = BoundaryEnforcer(&di_fn,boundary);
  BallCost cost_fn = BallCost(0.15,zeros<vec>(2));

  Problem problem;
  problem.trans_fn = &bounded_di_fn;
  problem.cost_fn = &cost_fn;
  problem.discount = 0.997;

  DIBangBangPolicy policy = DIBangBangPolicy(actions);

  simulate(x0,problem,policy,T,res);  
}
