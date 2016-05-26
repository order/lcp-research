#include <iostream>
#include <assert.h>
#include <armadillo>

#include "misc.h"
#include "simulate.h"

using namespace arma;

void add_to_simout(const mat & points,
		   const mat & actions,
		   const vec & costs,
		   uint t,
		   SimulationOutcome & outcome){
    
    outcome.points.slice(t) = points;
    outcome.actions.slice(t) = actions;
    outcome.costs.col(t) = costs;
}
void add_to_gain(const vec & costs,
		 double discount,
		 uint t,
		 vec & gain){
  gain += pow(discount,t) * costs;
}

void simulate(const mat & X0,
	      const Problem & problem,
	      const Policy & policy,
	      uint T,
	      SimulationOutcome & outcome,
	      vec & gain,
	      uint flag){

  mat points = X0;
  uint N = points.n_rows;
  uint D = points.n_cols;
  uint A = policy.get_action_dim();

  double discount = problem.discount;

  // Init recording structures
  if (flag & RECORD_OUTCOME){
    outcome.points = cube(N,D,T);
    outcome.actions = cube(N,A,T);
    outcome.costs = mat(N,T);
  }
  if (flag & RECORD_GAIN){
    gain = vec(N).fill(0);
  } 
  
  mat actions;
  vec costs;
  for(uint t = 0; t < T; t++){
    actions = policy.get_actions(points);
    costs = problem.cost_fn->get_costs(points,actions);

    // Record
    if (flag & RECORD_OUTCOME){
      add_to_simout(points,actions,costs,t,outcome);
    }    
    if (flag & RECORD_GAIN){
      add_to_gain(costs,discount,t,gain);
    }    

    if(t < T-1){
      points = problem.trans_fn->get_next_states(points,actions);
    }
  }
}

void simulate_outcome(const mat & x0,
		      const Problem & problem,
		      const Policy & policy,
		      uint T,
		      SimulationOutcome & outcome){
  vec dummy;
  simulate(x0,problem,policy,T,outcome,dummy,RECORD_OUTCOME);
}

void simulate_gain(const mat & x0,
		   const Problem & problem,
		   const Policy & policy,
		   uint T,
		   vec & gain){
  SimulationOutcome dummy;
  simulate(x0,problem,policy,T,dummy,gain,RECORD_GAIN);
}

void simulate_test(SimulationOutcome & res){
  /*
    Basic DI simulation with no arguments
   */
  uint T = 100;
  arma_rng::set_seed_random();
  mat x0 = 100*ones<mat>(1,2);
  
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

  simulate_outcome(x0,problem,policy,T,res);  
}
