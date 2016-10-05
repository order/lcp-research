#ifndef __Z_BASIS_INCLUDED__
#define __Z_BASIS_INCLUDED__

#include "discretizer.h"

#include <armadillo>
#include <set>
#include <vector>
#include "misc.h"

using namespace std;
using namespace arma;

double find_radius(const vec & dist,
                   uint target);
double find_radius(const vec & dist,
                   const bvec & mask,
                   uint target);

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

typedef vector<set<uint> > IndexPartition;
typedef IndexPartition::iterator IndexIterator;

IndexPartition voronoi_partition(const Points & points,
                                 const Points & centers);
set<uint> ball_indices(const Points & points,
                       const vec & center,
                       uint R);
void add_basis(IndexPartition & partition,
                         const set<uint> & basis);
sp_mat build_basis_from_partition(const IndexPartition &,uint);


class VoronoiBasis{
 public:
  VoronoiBasis(const Points & points);
  VoronoiBasis(const Points & points,
               const Points & centers);

  void add_center(const vec & center);
  sp_mat get_basis() const;

  Points m_points;
  Points m_centers;
  mat m_dist;

  uint n_dim;
  uint n_basis;
  uint n_points;
};

#endif
