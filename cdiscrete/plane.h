#ifndef __Z_PLANE_INCLUDED__
#define __Z_PLANE_INCLUDED__

#include "simulator.h"
#include "tri_mesh.h"

#define TWO_ACTIONS 2
#define NMAC_RADIUS 1

/*
 * This is a simulator for a 3D relative motion model for two planes.
 * NB: 4D if we have a previous advisory state (could be very small dim e.g. 5)
 */

class RelativePlanesSimulator : public TypedSimulator{
 public:
  RelativePlanesSimulator(const arma::mat & bbox,
			  const arma::mat &actions,
			  double noise_std = 0.1,
			  double step=0.01);
  arma::mat get_costs(const TypedPoints & points) const;
  arma::vec get_state_weights(const TypedPoints & points) const;
  arma::mat get_actions() const;
  TypedPoints next(const TypedPoints & points,
              const arma::vec & actions) const;
  arma::sp_mat transition_matrix(const TypedDiscretizer * disc,
				 const arma::vec & action,
				 bool include_oob) const;

  std::vector<arma::sp_mat> transition_blocks(const TypedDiscretizer * disc,
                                   uint num_samples=1) const;
  std::vector<arma::sp_mat> lcp_blocks(const TypedDiscretizer * disc,
                            const double gamma,
                            uint num_samples=1) const;
  arma::mat q_mat(const TypedDiscretizer * disc) const;

  uint num_actions() const;
  uint dim_actions() const;

  arma::mat get_bounding_box() const;
  
 protected:
  arma::mat m_actions;
  arma::mat m_bbox;
  double m_step;
  double m_noise_std;
  double m_damp;
};

#endif
