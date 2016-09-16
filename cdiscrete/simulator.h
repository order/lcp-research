#ifndef __Z_SIMULATE_INCLUDED__
#define __Z_SIMULATE_INCLUDED__

#include <armadillo>
#include "tri_mesh.h"

class Simulator{
 public:
  virtual mat get_costs(const Points & points) const = 0;
  virtual mat get_actions() const = 0;
  virtual Points next(const Points & points,
                      const vec & actions) const = 0;
  virtual uint num_actions() const = 0;
  virtual uint dim_actions() const = 0;
};

void saturate(Points & points,
              const uvec & idx,
              const mat & bbox);
void wrap(Points & points,
          const uvec & idx,
          const mat & bbox);

mat find_bounding_box(const TriMesh & mesh);

vec lp_norm_weights(const Points & points,
                    double p);

mat estimate_Q(const Points & points,
               const TriMesh & mesh,
               const Simulator * sim,
               const vec & values,
               double gamma,
               uint samples);

#endif
