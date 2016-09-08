#ifndef __Z_DI_INCLUDED__
#define __Z_DI_INCLUDED__

#include "mesh.h"

using namespace arma;
using namespace std;

#define SIM_STEP 0.02

// DI stuff
void add_di_bang_bang_curves(TriMesh & mesh,
                             const vec & lb,
                             const vec & ub,
                             uint num_curve_points);
Points double_integrator(const Points & points,
			 double a,double t);
mat build_di_costs(const Points & points);
vec build_di_state_weights(const Points & points);
sp_mat build_di_transition(const Points & points,
			   const TriMesh & mesh,
			   const vec & lb,
			   const vec & ub,
			   double action);

void saturate(Points & points, 
              const vec &lb,
              const vec &ub); // TODO: Should move to a more general file

void build_square_boundary(TriMesh & mesh,
			   const vec & lb,
			   const vec & ub); // TODO: Should move to a more general file

bool check(const sp_mat & A); // TODO: Should move to a more general file
#endif
