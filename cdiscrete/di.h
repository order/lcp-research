#ifndef __Z_DI_INCLUDED__
#define __Z_DI_INCLUDED__

#include "simulator.h"


using namespace arma;
using namespace std;

#define DOUBLE_INT_DIM 2

class DoubleIntegratorSimulator : public Simulator{
 public:
  DoubleIntegratorSimulator(const mat & bbox,
                            const mat &actions = vec{-1,1},
                            double noise_std = 0.1,
                            double step=0.01);
  
  mat get_costs(const Points & points) const;
  mat get_costs(const TypedPoints & points) const;
  
  vec get_state_weights(const Points & points) const;
  vec get_state_weights(const TypedPoints & points) const;
  
  mat get_actions() const;
  
  Points next(const Points & points,
              const vec & actions) const;
  TypedPoints next(const TypedPoints & points,
              const vec & actions) const;

  sp_mat transition_matrix(const Discretizer * disc,
                           const vec & action,
                           bool include_oob) const;
  sp_mat transition_matrix(const TypedDiscretizer * disc,
                           const vec & action) const;

  vector<sp_mat> transition_blocks(const Discretizer * disc,
                                   uint num_samples=1) const;
  vector<sp_mat> transition_blocks(const TypedDiscretizer * disc,
                                   uint num_samples=1) const;

  vector<sp_mat> lcp_blocks(const Discretizer * disc,
                            const double gamma,
                            uint num_samples=1) const;
  vector<sp_mat> lcp_blocks(const TypedDiscretizer * disc,
                            const double gamma,
                            uint num_samples=1) const;
  mat q_mat(const Discretizer * disc) const;
  mat q_mat(const TypedDiscretizer * disc) const;

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
