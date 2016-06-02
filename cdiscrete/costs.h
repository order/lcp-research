#ifndef __Z_COSTS_INCLUDED__
#define __Z_COSTS_INCLUDED__

#include <armadillo>

#include "misc.h"

using namespace arma;

//=================================================
// COSTS

class CostFunction{
 public:
  virtual double get_cost(const vec & point, const vec & actions) const = 0;
  virtual mat get_costs(const mat & points, const mat & actions) const = 0;
};

class BallCost : public CostFunction{
 public:
  BallCost(double radius, const vec & center);
  double get_cost(const vec & point, const vec & actions) const;
  mat get_costs(const mat & points, const mat & actions) const;

 protected:
  double _radius;
  rowvec _center;
};

#endif
