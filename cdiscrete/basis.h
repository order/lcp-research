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
vec gaussian(const Points & points,
             const vec & center,
             const mat & cov);
vec laplacian(const Points & points,
	      const vec & center,
	      const double bandwidth);

double find_radius(const vec & dist,
                   uint target);
double find_radius(const vec & dist,
                   const bvec & mask,
                   uint target);
mat dist_mat(const Points & A,
             const Points & B);

vec make_ball(const Points & points,
	      const vec & center,
	      const double radius);

sp_mat make_sample_basis(uint N,uint K);
sp_mat make_ball_basis(const Points & points,
                       const Points & centers,
                       uint R);
mat make_rbf_basis(const Points & points,
                   const Points & centers,
                   double bandwidth,
		   double cutoff_thresh=1e-5);

mat make_rbf_basis(const TypedPoints & points,
                   const Points & centers,
                   double bandwidth,
		   double cutoff_thresh=1e-5);

sp_mat make_radial_fourier_basis(const Points & points,
                              uint K, double max_freq);

sp_mat make_fourier_basis(const Points & points,
                          uint K, double max_freq);
sp_mat make_voronoi_basis(const Points & points,
                          const Points & centers);

sp_mat make_voronoi_basis(const TypedPoints & points,
			  const Points & cuts);


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

class TypedVoronoiBasis{
 public:
  TypedVoronoiBasis(const TypedPoints & points,
		    const Points & centers);
  sp_mat get_basis() const;

  TypedPoints m_typed_points;
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

vector<sp_mat> make_freebie_flow_bases_ignore_q(const sp_mat & value_basis,
                                                const vector<sp_mat> blocks);
vector<sp_mat> make_freebie_flow_bases(const sp_mat & value_basis,
                                       const vector<sp_mat> blocks,
                                       const mat & Q);
vector<sp_mat> balance_bases(const vector<sp_mat> initial_bases,
			     const vector<sp_mat> blocks);
vector<mat> make_raw_freebie_flow_bases(const mat & raw_value_basis,
                                        const vector<sp_mat> blocks,
                                        const mat & Q);
PLCP approx_lcp(const sp_mat & value_basis,
                const sp_mat & smoother,
                const vector<sp_mat> & blocks,
                const mat & Q,
                const bvec & free_vars,
		bool ignore_q);

////////////////////////////////////////////////////////////////////////////
// Variable resolution bases //
///////////////////////////////

class TabularVarResBasis{
  /*
   * Groups the discrete points into axis-parallel boxes, each with a uniform
   * value
   */
 public:
  TabularVarResBasis(const TypedPoints & points, mat bounds);
  sp_mat get_basis() const;
  bool can_split(uint basis_idx, uint dim_idx) const;
  uint split_basis(uint basis_idx, uint dim_idx);
  
  
  TypedPoints m_typed_points;
  uint n_bases;
  uint n_rows;
  uint n_cols;
  vector<vector<uint>> m_point_assign;
  vector<mat> m_basis_bounds;
};

umat box2points(const umat & bbox, uvec grid_size);
umat box2vert(const umat & bbox, uvec grid_size);
uvec box2width(const umat & bbox);
bool is_bbox(const umat & bbox);

class MultiLinearVarResBasis{
 public:
  MultiLinearVarResBasis(const uvec & grid_size);
  sp_mat get_basis() const;

  // TODO: implement with fn pointers
  vec get_cell_fn_reduced(const vec & x, int(*fn)(const * vec)) const;
  vec get_cell_fn(const vec & x, int(*fn)(const * vec)) const;

  // Make instance of the above fns
  vec get_cell_min(const vec & x) const;
  vec get_cell_avg(const vec & x) const;
  vec get_cell_var(const vec & x) const;
  vec get_cell_var_reduced(const vec & x) const;

  bool can_split(uint cell_idx, uint dim_idx) const;
  uint split_cell(uint cell_idx, uint dim_idx);

  void split_per_dimension(uint cell_idx, uvec split);

  uvec m_grid_size;
  uint m_num_oob;  // Fixed to one for now.
  vector<umat> m_cell_to_bbox;
};

#endif
