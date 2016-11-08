#ifndef __Z_BASIS_INCLUDED__
#define __Z_BASIS_INCLUDED__

#include "discretizer.h"

#include <armadillo>
#include <set>
#include <vector>
#include "misc.h"
#include "lcp.h"

using namespace std;
using namespace arma;

// TODO: non-isometric versions
vec gabor_wavelet(const vec & points,
                  const double center,
                  const double freq,
                  const double bandwidth,
                  const double shift);
vec gabor_wavelet(const Points & points,
                  const vec & center,
                  const vec & freq,
                  const double bandwidth,
                  const double shift);
vec gaussian(const vec & points,
             const double center,
             const double bandwidth);
vec gaussian(const Points & points,
             const vec & center,
             const double bandwidth);


double find_radius(const vec & dist,
                   uint target);
double find_radius(const vec & dist,
                   const bvec & mask,
                   uint target);
mat dist_mat(const Points & A,
             const Points & B);

sp_mat make_sample_basis(uint N,uint K);
sp_mat make_ball_basis(const Points & points,
                       const Points & centers,
                       uint R);
mat make_rbf_basis(const Points & points,
                   const Points & centers,
                   double bandwidth);
sp_mat make_radial_fourier_basis(const Points & points,
                              uint K, double max_freq);

sp_mat make_fourier_basis(const Points & points,
                          uint K, double max_freq);
sp_mat make_voronoi_basis(const Points & points,
                          const Points & centers);


class VoronoiBasis{
 public:
  VoronoiBasis(const Points & points);
  VoronoiBasis(const Points & points,
               const Points & centers);

  void add_center(const vec & center);
  void replace_last_center(const vec & center);
  
  uint count(uint k) const;
  uint count_last() const;
  uint min_count() const;

  sp_mat get_basis() const;

  Points m_points;
  Points m_centers;
  mat m_dist;

  uint n_dim;
  uint n_basis;
  uint n_points;
};

sp_mat make_basis(const string&,
                  const vector<double>&,
                  const Points &,
                  uint K);

LCP smooth_lcp(const sp_mat & smoother,
               const vector<sp_mat> & blocks,
               const mat & Q,
               const bvec & free_vars);

vector<sp_mat> make_freebie_flow_bases(const sp_mat & value_basis,
                                       const vector<sp_mat> blocks);

PLCP approx_lcp(const sp_mat & value_basis,
                const sp_mat & smoother,
                const vector<sp_mat> & blocks,
                const mat & Q,
                const bvec & free_vars);

#endif
