#ifndef __Z_DI_INCLUDED__
#define __Z_DI_INCLUDED__

#include "mesh.h"

using namespace arma;
using namespace std;

// DI stuff
void add_di_bang_bang_curves(TriMesh & mesh,
			     VertexHandle & v_zero,
			     VertexHandle & v_upper_left,
			     VertexHandle & v_lower_right,
			     uint num_curve_points);

Points double_integrator(const Points & points,
			 double a,double t);

mat build_di_costs(const Points & points);
vec build_di_state_weights(const Points & points);

#endif
