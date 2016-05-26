#ifndef __Z_TRANSFER_INCLUDED__
#define __Z_TRANSFER_INCLUDED__

#include <armadillo>
#include "discrete.h"
using namespace arma;


//=================================================
// ABSTRACT TRANSFER FUNCTIONS

// Abstract transfer function
class TransferFunction{
 public:
  virtual mat get_next_states(const mat & points,
			      const mat & actions) const = 0;

  // Single point, single action
  vec get_next_state(const vec & points,
		     const vec & action) const;

  // Many points, single action
  mat get_next_states(const mat & points,
		      const vec & actions) const;

};
//=================================================
// SPECIFIC TRANSFER FUNCTIONS

class DoubleIntegrator : public TransferFunction{
 public:
  DoubleIntegrator(double step_size,
		   uint num_steps,
		   double damping,
		   double jitter);
  mat get_next_states(const mat & points, const mat & actions) const;
  
 private:
  double _step_size;
  uint _num_steps;
  double _damping;
  double _jitter;
};

//=================================================
// BOUNDARY AND ENFORCEMENT

struct Boundary{
  vec low;
  vec high;
};

class BoundaryEnforcer : public TransferFunction{
 public:
  BoundaryEnforcer(TransferFunction * trans_fn_ptr, const Boundary & boundary);
  BoundaryEnforcer(TransferFunction * trans_fn_ptr, const RegGrid & boundary);
  mat get_next_states(const mat & points, const mat & actions) const;
  vec get_next_state(const vec & point, const vec & action) const;
 protected:
  TransferFunction * _trans_fn_ptr;
  Boundary _boundary;
};

#endif
