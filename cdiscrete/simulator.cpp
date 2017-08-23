#include "simulator.h"
#include "misc.h"
#include <assert.h>

//////////////////////////////////////////
// Functions for boundary remapping
void saturate(Points & points,
              const uvec &indices,
              const mat &bbox){
  /*
    Project points back onto bounding box.
    Think "perfectly inelastic collision"

    "indices" are the dimensions that we want 
    to saturate along

    bbox is a Dx2 matrix [lb, ub]
  */  
  uint D = points.n_cols;
  assert(D == bbox.n_rows);
  assert(2 == bbox.n_cols);
  assert(indices.n_elem <= D);

  uvec mask;
  uint I = indices.n_elem;
  vec lb = bbox.col(0);
  vec ub = bbox.col(1);
  for(uint i = 0; i < I; i++){
    uint idx = indices(i); // Which dimension
    uvec col_idx = uvec{idx};
    
    assert(is_finite(bbox.row(idx)));
    assert(lb(idx) < ub(idx));
    
    mask = find(points.col(idx) < lb(idx) + PRETTY_SMALL);
    points(mask,col_idx).fill(lb(idx) + PRETTY_SMALL);
    
    mask = find(points.col(idx) > ub(idx) - PRETTY_SMALL);
    points(mask,col_idx).fill(ub(idx) - PRETTY_SMALL);
  }
}


void wrap(Points & points,
          const uvec &indices,
          const mat &bbox){
  /*
    Wrap points back onto bounding box.
    Think "torus"
  */
  
  uint D = points.n_cols;
  assert(D == bbox.n_rows);
  assert(2 == bbox.n_cols);
  assert(D >= indices.n_elem);

  uvec mask;
  uint I = indices.n_elem;
  vec lb = bbox.col(0);
  vec ub = bbox.col(1);
  uint idx;
  for(uint i = 0; i < I; i++){
    idx = indices(i);
    assert(is_finite(bbox.row(idx)));
    assert(lb(idx) < ub(idx));
    
    points.col(idx) -= lb(idx); // p - l
    points.col(idx) = vec_mod(points.col(idx),
                              ub(idx) - lb(idx));
    points.col(idx) += lb(idx);    // (p-l) % (u-l) + l
    
    assert(not any(points.col(idx) > ub(idx)));
    assert(not any(points.col(idx) < lb(idx)));
  }   
}

uvec out_of_bounds(const Points & points,
                   const uvec & indices,
                   const mat & bbox){
  uint N = points.n_rows;
  uint D = points.n_cols;
  uint I = indices.n_elem;
  assert(D == bbox.n_rows);
  assert(2 == bbox.n_cols);
  assert(D >= indices.n_elem);

  uvec mask;
  vec lb = bbox.col(0);
  vec ub = bbox.col(1);
  
  uint idx;
  vector<uword> oob_vec;
  for(uint i = 0; i < I; i++){
    idx = indices(i);
    mask = find(points.col(idx) < lb(idx));
    oob_vec.insert(oob_vec.end(),mask.begin(),mask.end());
    
    mask = find(points.col(idx) > ub(idx));
    oob_vec.insert(oob_vec.end(),mask.begin(),mask.end());

    mask = find_nonfinite(points.col(idx));
    oob_vec.insert(oob_vec.end(),mask.begin(),mask.end());
  }

  uvec oob = uvec(oob_vec);
  return unique(oob);
}

vec lp_norm_weights(const Points & points,
                    double p){
  vec w = lp_norm(points,p,1);
  assert(w.n_elem == points.n_rows);
  return w / accu(w); // Normalize
}
 
mat estimate_Q(const Points & points,
               const Discretizer * disc,
               const Simulator * sim,
               const vec & values,
               double gamma,
               int steps,
               uint samples){
  assert(steps >= 0);
  uint N = points.n_rows;
  uint D = points.n_cols;
  
  mat Q = sim->get_costs(points);
  assert(N == Q.n_rows);
  uint A = Q.n_cols;
  
  mat actions = sim->get_actions();
  assert(A == actions.n_rows);
  uint Ad = actions.n_cols;

  Points next_points;
  vec next_v;
  mat next_Q;  
  for(uint a = 0; a < A; a++){
    vec action = actions.row(a).t();
    next_v = zeros<vec>(N);
    if(steps > 0){
      next_Q = zeros<mat>(N,A);
    }
    
    // Averages over samples
    for(uint i = 0; i < samples; i++){
      Points next_points = sim->next(points,action);
      if(steps == 0){
        next_v += disc->interpolate(next_points,values);
      }
      else{
        next_Q += estimate_Q(next_points,disc,sim,
                             values,gamma,
                             steps-1,
                             samples);
      }
    }
    if(steps > 0){
      next_v = min(next_Q,1);
    }
    next_v /= (double) samples;
    Q.col(a) += gamma * next_v;
  }
  return Q;
}

mat estimate_Q(const TypedPoints & points,
               const TypedDiscretizer * disc,
               const TypedSimulator * sim,
               const vec & values,
               double gamma,
               int steps,
               uint samples){
  assert(steps >= 0);
  uint N = points.n_rows;
  uint D = points.n_cols;
  
  mat Q = sim->get_costs(points);
  assert(N == Q.n_rows);
  uint A = Q.n_cols;
  
  mat actions = sim->get_actions();
  assert(A == actions.n_rows);
  uint Ad = actions.n_cols;

  vec next_v;
  mat next_Q;  
  for(uint a = 0; a < A; a++){
    vec action = actions.row(a).t();
    next_v = zeros<vec>(N);
    if(steps > 0){
      next_Q = zeros<mat>(N,A);
    }
    
    // Averages over samples
    for(uint i = 0; i < samples; i++){
      TypedPoints next_points = sim->next(points, action);
      if(steps == 0){
        next_v += disc->interpolate(next_points, values);
      }
      else{
        next_Q += estimate_Q(next_points,disc,sim,
                             values,gamma,
                             steps-1,
                             samples);
      }
    }
    if(steps > 0){
      next_v = min(next_Q,1);
    }
    next_v /= (double) samples;
    Q.col(a) += gamma * next_v;
  }
  return Q;
}
