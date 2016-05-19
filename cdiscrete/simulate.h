#ifndef __POLICY_INCLUDED__
#define __POLICY_INCLUDED__

#include <armadillo>
using namespace arma;

//=================================================
// POLICIES

// Abstract policy class
class Policy{
 public:
  virtual vec get_actions(const mat & points);
};

// Bang-bang policy for 2D (x,v) double integrator
class DIBangBangPolicy : public Policy{
 public:
  DIBangBangPolicy(const mat & actions);
  uvec get_action_indices(const mat & points);
  vec get_actions(const mat & points);
 private:
  mat _actions;
  uint _n_actions;
};
//=================================================
// TRANSFER FUNCTIONS

// Abstract transfer function
class TransferFunction{
 public:
  virtual mat get_next_states(const mat & points, const mat & actions);
};

class DoubleIntegrator : public TransferFunction{
 public:
  DoubleIntegrator(double step_size,
		   uint num_steps,
		   double damping,
		   double jitter);
  mat get_next_states(const mat & points, const mat & actions);
  
 private:
  double _step_size;
  uint _num_steps;
  double _damping;
  double _jitter;
};

#endif
