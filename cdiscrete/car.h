#ifndef __Z_CAR_INCLUDED__
#define __Z_CAR_INCLUDED__

#include "tet_mesh.h"

using namespace arma;
using namespace std;

#define SIM_STEP 0.02

/*
  Physics for the Reeds-Shepp or Dubbins
car
*/

Points car(const Points & points,
           double u1, double u2);
mat build_car_costs(const Points & points,
                    uint num_actions, double oob_cost);
vec build_car_state_weights(const Points & points);
sp_mat build_car_transition(const Points & points,
                            const tet_mesh::TetMesh & mesh,
                            double u1, double u2,
                            double oob_self_prob);
#endif
