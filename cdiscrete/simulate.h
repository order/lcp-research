#ifndef __Z_SIMULATE_INCLUDED__
#define __Z_SIMULATE_INCLUDED__

#include <armadillo>
#include "transfer.h"
#include "costs.h"
#include "policy.h"

// Threshold for components to stop simulating
#define SIMTHRESH 0.05

#define RECORD_GAIN    1
#define RECORD_FINAL   2
#define RECORD_OUTCOME 4

using namespace arma;

//===================================================
// PROBLEM DESCRIPTION
struct Problem{
  mat actions;
  TransferFunction * trans_fn;
  CostFunction * cost_fn;
  double discount;
};
void delete_problem(Problem * problem);

//==================================================
// SIMULATOR

struct SimulationOutcome{
  cube points;
  cube actions;
  mat costs;  
};

void add_to_simout(const mat & points,
		   const mat & actions,
		   const vec & costs,
		   uint t,
		   SimulationOutcome & outcome);
void add_to_gain(const vec & costs,
		 double discount,
		 uint t,
		 vec & gain);
double simulate_single(const vec & x0,
	      const Problem & problem,
	      const Policy & policy,
	      uint T,
	      vec & final_point);

/*
void simulate(const mat & x0,
	      const Problem & problem,
	      const Policy & policy,
	      uint T,
	      SimulationOutcome & outcome,
	      vec & gain,
	      uint flag);
*/
/*
void simulate_outcome(const mat & x0,
	      const Problem & problem,
	      const Policy & policy,
	      uint T,
	      SimulationOutcome & outcome);
*/
/*
void simulate_gain(const mat & x0,
		   const Problem & problem,
		   const Policy & policy,
		   uint T,
		   vec & gain,
		   mat & final_x);
*/
//void simulate_test(SimulationOutcome & res);

#endif
