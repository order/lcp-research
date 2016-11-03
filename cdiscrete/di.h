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
  vec get_state_weights(const Points & points) const;
  mat get_actions() const;
  Points next(const Points & points,
              const vec & actions) const;
  sp_mat transition_matrix(const Discretizer * disc,
                           const vec & action,
                           bool include_oob) const;

  vector<sp_mat> transition_blocks(const Discretizer * disc) const;
  vector<sp_mat> lcp_blocks(const Discretizer * disc,
			    const double gamma) const;
  mat q_mat(const Discretizer * disc) const;

  void add_bang_bang_curve(tri_mesh::TriMesh & mesh,
                           uint num_curve_points) const;

  uint num_actions() const;
  uint dim_actions() const;

  mat get_bounding_box() const;
  
 protected:
  mat m_actions;
  mat m_bbox;
  double m_step;
  double m_noise_std;
  double m_damp;
};

tri_mesh::TriMesh generate_initial_mesh(double angle,
                                        double length,
                                        const mat & bbox);

#endif
