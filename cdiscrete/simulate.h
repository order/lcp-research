#ifndef __Z_SIMULATE_INCLUDED__
#define __Z_SIMULATE_INCLUDED__

#include <armadillo>
#include "transfer.h"
#include "costs.h"
#include "policy.h"

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

void simulate(const mat & x0,
	      const Problem & problem,
	      const Policy & policy,
	      uint T,
	      SimulationOutcome & outcome);

void simulate_test(SimulationOutcome & res);

#endif
