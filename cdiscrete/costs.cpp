#include "costs.h"

BallCosts::BallCosts(double radius, const vec & center){
  _radius = radius;
  _center = conv_to<rowvec>::from(center);
}

vec BallCosts::get_costs(const mat & points, const mat & actions) const{
  uint N = points.n_rows;
  vec c = vec(N);
  
  uvec in_ball = dist(points,_center) <= _radius;
  
  c.rows(find(in_ball == 1)).fill(0);
  c.rows(find(in_ball == 0)).fill(1);
  return c;
}
