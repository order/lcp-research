#ifndef __Z_HILLCAR_INCLUDED__
#define __Z_HILLCAR_INCLUDED__

#include "simulator.h"
#include "tri_mesh.h"

using namespace arma;
using namespace std;

namespace hillcar{
  static const int GRAVITY = 9.8;
  static const mat HILLCAR_ACTIONS = vec{-1.5,1.5};
  static const double HILLCAR_GAMMA = 0.999;
  static const mat HILLCAR_BBOX = mat{{-6,2},{-5,5}};

  class HillcarSimulator : public Simulator{
  public:
    HillcarSimulator(const mat & bbox,
                     const mat &actions = vec{-1,1},
                     double noise_std = 0.25,
                     double step=0.01);
    mat get_costs(const Points & points) const;
    vec get_state_weights(const Points & points) const;
    mat get_actions() const;
    void enforce_boundary(Points & points) const;
    Points next(const Points & points,
                const vec & actions) const;
    sp_mat transition_matrix(const Discretizer *,
                             const vec & action,
                             bool include_oob) const;

    uint num_actions() const;
    uint dim_actions() const;

    mat get_bounding_box() const;
  
  protected:
    mat m_actions;
    mat m_bbox;
    double m_step;
    double m_noise_std;

    static const int HILLCAR_DIM = 2;
    static const int X_VAR = 0;
    static const int V_VAR = 1;
  };

  vec triangle_wave(vec x, double P, double A);
  vec triangle_slope(vec x);
}
#endif
