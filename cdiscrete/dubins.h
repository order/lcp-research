#ifndef __Z_CAR_INCLUDED__
#define __Z_CAR_INCLUDED__

#include "simulator.h"
#include "discretizer.h"
#include "tet_mesh.h"

#include <armadillo>

using namespace arma;
using namespace std;

/*
  Physics for the Reeds-Shepp or Dubbins
car
*/
class DubinsCarSimulator : public Simulator{
 public:
  DubinsCarSimulator(const mat &actions,
                     double noise_std = 0.25,
                     double step=0.01);
  mat get_costs(const Points & points) const;
  vec get_state_weights(const Points & points) const;
  mat get_actions() const;
  Points next(const Points & points,
              const vec & actions) const;
  sp_mat transition_matrix(const Discretizer *,
                           const vec & action,
                           bool include_oob) const;

  uint num_actions() const;
  uint dim_actions() const;
  
 protected:
  mat m_actions;
  double m_step;
  double m_noise_std;

  static const int DUBINS_DIM = 3;
  static const int DUBINS_ACTION_DIM = 2;
  static const int X_VAR = 0;
  static const int V_VAR = 1;
  static const int Z_VAR = 2;
};
#endif
