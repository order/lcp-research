#ifndef __Z_HILLCAR_INCLUDED__
#define __Z_HILLCAR_INCLUDED__

#include "simulator.h"
#include "tri_mesh.h"

using namespace arma;
using namespace std;

#define GRAVITY 9.8

class HillcarSimulator : public Simulator{
 public:
  HillcarSimulator(const mat & bbox,
                            const mat &actions = vec{-1,1},
                            double noise_std = 0.1,
                            double step=0.01);
  mat get_costs(const Points & points) const;
  mat get_actions() const;
  void enforce_boundary(Points & points) const;
  Points next(const Points & points,
              const vec & actions) const;
  sp_mat transition_matrix(const TriMesh & mesh,
                           const vec & action) const;

  uint num_actions() const;
  uint dim_actions() const;

  mat get_bounding_box() const;
  
 protected:
  mat m_actions;
  mat m_bbox;
  double m_step;
  double m_noise_std;

  static const int X_VAR = 0;
  static const int V_VAR = 1;
};

vec triangle_wave(vec x, double P, double A);
vec triangle_slope(vec x);

#endif
