#ifndef __Z_DI_INCLUDED__
#define __Z_DI_INCLUDED__

#include "simulator.h"
#include "tri_mesh.h"

using namespace arma;
using namespace std;

#define TWO_ACTIONS 2
#define NMAC_RADIUS 1

/*
 * This is a simulator for a 3D relative motion model for two planes.
 * NB: 4D if we have a previous advisory state (could be very small dim e.g. 5)
 */

class RelativePlanesSimulator : public Simulator{
 public:
  RelativePlanesSimulator(const mat & bbox,
			  const mat &actions,
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

#endif
