#ifndef __Z_TRANSFER_INCLUDED__
#define __Z_TRANSFER_INCLUDED__

#include <armadillo>
using namespace arma;

//=================================================
// TRANSFER FUNCTIONS

// Abstract transfer function
class TransferFunction{
 public:
  virtual mat get_next_states(const mat & points,
			      const mat & actions) const = 0;
  vec get_next_state(const vec & points,
			     const vec & action) const;
};

class DoubleIntegrator : public TransferFunction{
 public:
  DoubleIntegrator(double step_size,
		   uint num_steps,
		   double damping,
		   double jitter);
  mat get_next_states(const mat & points, const mat & actions) const;
  vec get_next_state(const vec & point, const vec & action) const;
  
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
  BoundaryEnforcer(TransferFunction * trans_fn_ptr, Boundary & boundary);
  mat get_next_states(const mat & points, const mat & actions) const;
  vec get_next_state(const vec & point, const vec & action) const;
 protected:
  TransferFunction * _trans_fn_ptr;
  Boundary _boundary;
};

#endif
