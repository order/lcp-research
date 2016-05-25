#include <assert.h>

#include "costs.h"

BallCost::BallCost(double radius, const vec & center){
  _radius = radius;
  _center = conv_to<rowvec>::from(center);
}

mat BallCost::get_costs(const mat & points, const mat & actions) const{
  uint N = points.n_rows;
  uint A = actions.n_rows;
  mat c = mat(N,A);

  uint D = _center.n_elem;
  assert(D == points.n_cols);
    
  uvec in_ball = dist(points,_center) <= _radius;
  
  c.rows(find(in_ball == 1)).fill(0);
  c.rows(find(in_ball == 0)).fill(1);
  return c;
}
