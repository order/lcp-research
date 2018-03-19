#ifndef __Z_DI_INCLUDED__
#define __Z_DI_INCLUDED__

#include "tri_mesh.h"
void add_bang_bang_curve(const DoubleIntegratorSimulator & sim,
			 tri_mesh::TriMesh & mesh,
			 uint num_curve_points) const;

tri_mesh::TriMesh generate_initial_mesh(double angle,
                                        double length,
                                        const arma::mat & bbox);

#endif
