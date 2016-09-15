#include "car.h"

Points car(const Points & points,
           double u1, double u2){
  assert(TET_NUM_DIM == points.n_cols);
  assert(points.is_finite());

  Points new_points = Points(points);
  
  new_points.col(0) += SIM_STEP * u1 * cos(points.col(2)); // x
  new_points.col(1) += SIM_STEP * u1 * sin(points.col(2)); // y
  new_points.col(2) += SIM_STEP * u1*u2; // theta

  // Angle wrap
  uvec col_idx = uvec{2};
  uvec row_idx = find(new_points.col(2) > datum::pi);
  new_points(row_idx,col_idx) -= 2*datum::pi;
  row_idx = find(new_points.col(2) < -datum::pi);
  new_points(row_idx,col_idx) += 2*datum::pi;

  assert(all(new_points.col(2) <= datum::pi));
  assert(all(new_points.col(2) >= -datum::pi));

  // May be out of bounds
  return new_points;
}

mat build_car_costs(const Points & points,
                       uint num_actions, double oob_cost){
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(TET_NUM_DIM == D);
  assert(points.is_finite());

  vec l2_norm = vec(N);
  l2_norm =  sqrt(sum(pow(points,2),1));
  assert(all(l2_norm >= 0));

  vec cost = min(ones<vec>(N),max(zeros<vec>(N),l2_norm - 0.5));
  mat costs = repmat(cost,1,num_actions);
  return costs;
}
vec build_car_state_weights(const Points & points){
  uint N = points.n_rows;
  assert(points.is_finite());
  
  vec weight = vec(N);
  weight = sqrt(sum(pow(points,2),1));
  return weight / sum(weight);
}
sp_mat build_car_transition(const Points & points,
                            const TetMesh & mesh,
                            double u1, double u2,
                            double oob_self_prob){
  assert(0 <= oob_self_prob);
  assert(1 >= oob_self_prob);
  assert(points.is_finite());

  uint N = points.n_rows;
  Points p_next = car(points,u1,u2);
  ElementDist ret = mesh.points_to_element_dist(p_next);
  
  // Final row is the OOB row
  assert(size(N+1,N) == size(ret));

  // Append OOB->OOB deterministic transition
  ElementDist final_col = ElementDist(N+1,1);
  final_col(N,0) = oob_self_prob;
  ret = join_horiz(ret,final_col);
  assert(size(N+1,N+1) == size(ret));

  return ret;
}
