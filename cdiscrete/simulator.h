#ifndef __Z_SIMULATE_INCLUDED__
#define __Z_SIMULATE_INCLUDED__

#include <armadillo>
#include "discretizer.h"
#include "misc.h"

class Simulator{
 public:
  virtual mat get_costs(const Points & points) const = 0;
  virtual vec get_state_weights(const Points & points) const = 0;
  virtual mat get_actions() const = 0;
  virtual Points next(const Points & points,
                      const vec & actions) const = 0;
  virtual sp_mat transition_matrix(const Discretizer *,
                                   const vec & action,
                                   bool include_oob) const = 0;
  virtual uint num_actions() const = 0;
  virtual uint dim_actions() const = 0;
};

void saturate(Points & points,
              const uvec & idx,
              const mat & bbox);
void wrap(Points & points,
          const uvec & idx,
          const mat & bbox);

uvec out_of_bounds(const Points & points,
                   const uvec & idx,
                   const mat & bbox);

mat find_bounding_box(const Discretizer * disc);

vec lp_norm_weights(const Points & points,
                    double p);

mat estimate_Q(const Points & points,
               const Discretizer * mesh,
               const Simulator * sim,
               const vec & values,
               double gamma,
               uint samples);

#endif
