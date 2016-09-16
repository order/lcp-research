#ifndef __Z_DI_INCLUDED__
#define __Z_DI_INCLUDED__

#include "simulator.h"
#include "tri_mesh.h"

using namespace arma;
using namespace std;

class DoubleIntegratorSimulator : public Simulator{
 public:
  DoubleIntegratorSimulator(const mat & bbox,
                            const mat &actions = vec{-1,1},
                            double noise_std = 0.1,
                            double step=0.01);
  mat get_costs(const Points & points) const;
  mat get_actions() const;
  Points next(const Points & points,
              const vec & actions) const;
  sp_mat transition_matrix(const TriMesh & mesh,
                           const vec & action) const;

  void add_bang_bang_curve(TriMesh & mesh,
                           uint num_curve_points) const;

  uint num_actions() const;
  uint dim_actions() const;

  mat get_bounding_box() const;
  
 protected:
  mat m_actions;
  mat m_bbox;
  double m_step;
  double m_noise_std;
};
#endif
