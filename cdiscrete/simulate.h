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


//==================================================
// SIMULATOR

struct SimulationOutcome{
  cube points;
  cube actions;
  mat costs;  
};

void simulate(const mat & x0,
	      const TransferFunction & trans_fn,
	      const CostFunction & cost_fn, 
	      const Policy & policy,
	      uint T,
	      SimulationOutcome & outcome);

#endif
