#ifndef __Z_BASIS_INCLUDED__
#define __Z_BASIS_INCLUDED__

#include "discretizer.h"

#include <armadillo>


using namespace std;
using namespace arma;

mat make_ring_basis(const Points & points,
                    uint S);

sp_mat make_sample_basis(uint N,uint K);
sp_mat make_ball_basis(const Points & points,
                       const Points & centers,
                       uint R);
mat make_rbf_basis(const Points & points,
                   const Points & centers,
                   double bandwidth);
// Make sparse
mat make_grid_basis(const Points & points,
                    const mat & bbox,
                    uint X, uint Y);

// Make sparse
mat make_voronoi_basis(const Points & points,
                       const Points & centers);

mat make_radial_fourier_basis(const Points & points,
                              uint K, double max_freq);

#endif
