#ifndef __Z_COSTS_INCLUDED__
#define __Z_COSTS_INCLUDED__

#include <armadillo>

#include "misc.h"

using namespace arma;

//=================================================
// COSTS

class CostFunction{
 public:
  virtual vec get_costs(const mat & points, const mat & actions) const = 0;
};

class BallCost : public CostFunction{
 public:
  BallCost(double radius, const vec & center);
  vec get_costs(const mat & points, const mat & actions) const;

 protected:
  double _radius;
  rowvec _center;
};

#endif
