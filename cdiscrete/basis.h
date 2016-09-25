#ifndef __Z_BASIS_INCLUDED__
#define __Z_BASIS_INCLUDED__

#include "discretizer.h"

#include <armadillo>


using namespace std;
using namespace arma;

mat make_ring_basis(const Points & points,
                    uint S);
vec make_spike_basis(const Points & points,
                     const vec & center);
mat make_grid_basis(const Points & points,
                    const mat & bbox,
                    uint X, uint Y);

mat make_voronoi_basis(const Points & points,
                       const Points & centers);

mat make_radial_fourier_basis(const Points & points,
                              uint K, double max_freq);

#endif
