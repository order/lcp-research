#ifndef __Z_COSTS_INCLUDED__
#define __Z_COSTS_INCLUDED__

#include <armadillo>
using namespace arma;

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

#endif
