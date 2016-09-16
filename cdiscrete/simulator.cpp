#include "simulator.h"
#include "misc.h"

//////////////////////////////////////////
// Functions for boundary remapping
void saturate(Points & points,
              const uvec &idx,
              const mat &bbox){
  /*
    Project points back onto bounding box.
    Think "perfectly inelastic collision"
  */  
  uint D = points.n_cols;
  assert(D == bbox.n_rows);
  assert(2 == bbox.n_cols);
  assert(D >= idx.n_elem);

  uvec mask;
  uvec col_idx;
  uint I = idx.n_elem;
  for(uint i = 0; i < I; i++){
    col_idx = uvec({i});
    for(uint j = 0; j < 2; j++){
      if(j == 0)
        mask = find(points.col(i) < bbox(i,j));
      else
        mask = find(points.col(i) > bbox(i,j));
      points(mask,col_idx).fill(bbox(i,j));
    }
  }
}

void wrap(Points & points,
          const uvec &idx,
          const mat &bbox){
  /*
    Wrap points back onto bounding box.
    Think "torus"
  */
  
  uint D = points.n_cols;
  assert(D == bbox.n_rows);
  assert(2 == bbox.n_cols);
  assert(D >= idx.n_elem);

  uvec mask;
  uvec col_idx;
  uint I = idx.n_elem;
  vec lb = bbox.col(0);
  vec ub = bbox.col(1);
  for(uint i = 0; i < I; i++){
    assert(is_finite(bbox.row(i)));
    assert(lb(i) < ub(i));
           points.col(i) -= lb(i); // p - l
           points.col(i) -= floor(points.col(i)
                                  / (ub(i) - lb(i))); // (p-l) % (u-l)
           points.col(i) += lb(i);    // (p-l) % (u-l) + l
    
           assert(not any(points.col(i) > ub(i)));
           assert(not any(points.col(i) < lb(i)));
  }   
}

vec lp_norm_weights(const Points & points,
                    double p){
  vec w = lp_norm(points,p,1);
  assert(w.n_elem == points.n_rows);
  return w / accu(w); // Normalize
}
 
mat estimate_Q(const Points & points,
               const TriMesh & mesh,
               const Simulator * sim,
               const vec & values,
               double gamma,
               uint samples){
  uint N = points.n_rows;
  uint D = points.n_cols;
  
  mat Q = sim->get_costs(points);
  assert(N == Q.n_rows);
  uint A = Q.n_cols;
  
  mat actions = sim->get_actions();
  assert(A == actions.n_rows);
  uint Ad = actions.n_cols;

  Points next_points;
  vec interp_v;
  
  for(uint a = 0; a < A; a++){
    vec action = actions.row(a).t();
    interp_v = zeros<vec>(N);
    // Averages over samples
    for(uint i = 0; i < samples; i++){
      Points next_points = sim->next(points,action);
      interp_v += mesh.interpolate(next_points,values);
    }
    interp_v /= (double) samples;
    Q.col(a) += gamma * interp_v;
  }
  return Q;
}
