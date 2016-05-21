#ifndef __POLICY_INCLUDED__
#define __POLICY_INCLUDED__

#include <armadillo>
using namespace arma;

//=================================================
// POLICIES

// Abstract policy class
class Policy{
 public:
  virtual vec get_actions(const mat & points) const = 0;
  virtual uint get_action_dim() const = 0;
};

// Bang-bang policy for 2D (x,v) double integrator
class DIBangBangPolicy : public Policy{
 public:
  DIBangBangPolicy(const mat & actions);
  uvec get_action_indices(const mat & points) const;
  vec get_actions(const mat & points) const;
  uint get_action_dim() const;
 private:
  mat _actions;
  uint _n_actions;
};

//=================================================
// TRANSFER FUNCTIONS

// Abstract transfer function
class TransferFunction{
 public:
  virtual mat get_next_states(const mat & points,
			      const mat & actions) const = 0;
};

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
  BoundaryEnforcer(TransferFunction * trans_fn_ptr, Boundary & boundary);
  mat get_next_states(const mat & points, const mat & actions) const;
 protected:
  TransferFunction * _trans_fn_ptr;
  Boundary _boundary;
};

//=================================================
// COSTS

class CostFunction{
 public:
  virtual vec get_costs(const mat & points, const mat & actions) const = 0;
};

class BallCosts : public CostFunction{
 public:
  BallCosts(double radius, const vec & center);
  vec get_costs(const mat & points, const mat & actions) const;

 protected:
  double _radius;
  rowvec _center;
};

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
