#ifndef __Z_POLICY_INCLUDED__
#define __Z_POLICY_INCLUDED__

#include <armadillo>
using namespace arma;

//=================================================
// POLICIES				      

// Abstract policy class
class Policy{
 public:
  vec get_action(const vec & point) const;
  virtual mat get_actions(const mat & points) const = 0;
  virtual uint get_action_dim() const = 0;
};

class DiscretePolicy : public Policy{
 public:
  virtual uvec get_action_indices(const mat & points) const = 0;
  uint get_action_index(const vec & point) const;
};

// Bang-bang policy for 2D (x,v) double integrator
class DIBangBangPolicy : public DiscretePolicy{
 public:
  DIBangBangPolicy(const mat & actions);
  uvec get_action_indices(const mat & points) const;
  mat get_actions(const mat & points) const;
  uint get_action_dim() const;
 private:
  mat _actions;
  uint _n_actions;
};

// Bang-bang policy for 2D (x,v) double integrator
class DILinearPolicy : public DiscretePolicy{
 public:
  DILinearPolicy(const mat & actions);
  uvec get_action_indices(const mat & points) const;
  mat get_actions(const mat & points) const;
  uint get_action_dim() const;
 private:
  mat _actions;
  uint _n_actions;
};

// Bang-bang policy for 2D (x,v) double integrator
class HillcarPolicy : public DiscretePolicy{
 public:
  HillcarPolicy(const mat & actions);
  uvec get_action_indices(const mat & points) const;
  mat get_actions(const mat & points) const;
  uint get_action_dim() const;
 private:
  mat _actions;
  double _A,_B;
  uint _n_actions;
};


#endif
