#ifndef __Z_SIMULATE_INCLUDED__
#define __Z_SIMULATE_INCLUDED__

#include <armadillo>
#include "transfer.h"
#include "costs.h"
#include "policy.h"

#define RECORD_GAIN    1
#define RECORD_OUTCOME 2

using namespace arma;

//===================================================
// PROBLEM DESCRIPTION
struct Problem{
  mat actions;
  TransferFunction * trans_fn;
  CostFunction * cost_fn;
  double discount;
};


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

void simulate(const mat & x0,
	      const Problem & problem,
	      const Policy & policy,
	      uint T,
	      SimulationOutcome & outcome,
	      vec & gain,
	      uint flag);

void simulate_outcome(const mat & x0,
	      const Problem & problem,
	      const Policy & policy,
	      uint T,
	      SimulationOutcome & outcome);

void simulate_gain(const mat & x0,
		   const Problem & problem,
		   const Policy & policy,
		   uint T,
		   vec & gain);

void simulate_test(SimulationOutcome & res);

#endif