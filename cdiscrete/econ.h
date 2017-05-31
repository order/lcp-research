#ifndef __Z_ECON_INCLUDED__
#define __Z_ECON_INCLUDED__

#include "simulator.h"
#include "tri_mesh.h"

using namespace arma;
using namespace std;

class StockSimulator : public Simulator{
 public:
  StockSimulator(const mat & bbox, double step);
  mat get_actions() const;
  mat get_costs(const Points & points) const;
  vec get_state_weights(const Points & points) const;
  Points next(const Points & points,
              const vec & actions) const;
  sp_mat transition_matrix(const Discretizer * disc,
                           const vec & action,
                           bool include_oob) const;

  vector<sp_mat> transition_blocks(const Discretizer * disc,
                                   uint num_samples=1) const;
  vector<sp_mat> lcp_blocks(const Discretizer * disc,
                            const double gamma,
                            uint num_samples=1) const;
  mat q_mat(const Discretizer * disc) const;

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
