#ifndef __Z_GRID_INCLUDED__
#define __Z_GRID_INCLUDED__

#include <armadillo>
#include <climits>
#include <map>

#include "discretizer.h"
#include "misc.h"
#include "points.h"

using namespace std;
using namespace arma;

uvec c_order_stride(const uvec & points_per_dim);
uvec c_order_cell_shift(const uvec & points_per_dim);

class Coords{
 public:
  static const uint OOB_TYPE = 1;

  Coords(const umat & coords);
  Coords(const umat & coords, const TypeRegistry & reg);
  Coords(const TypedPoints & points);

  bool check(const uvec & grid_size) const;
  bool _coord_check(const uvec & grid_size, const umat & coords) const;
  bool _type_reg_check(const TypeRegistry & registry,
		       const umat & coords) const;
  umat _indicies_to_coords(const uvec & grid_size,
			     const uvec & indices) const;
  uvec _coords_to_indices(const uvec & grid_size,
			  const Coords & coords) const;
  void _mark(const uvec & indices, uint coord_type);
  void _mark(const TypeRegistry & reg);
  TypeRegistry _find_oob(const uvec & grid_size) const;

  uint num_coords() const;
  uint num_spatial() const;
  uint num_special() const;
  uint max_spatial_index(const uvec & grid_size) const;

  uvec get_indices(const uvec & grid_size) const;
  
  umat m_coords;
  uint n_rows;
  uint n_dim;
  TypeRegistry m_reg;
};

class UniformGrid : public TypedDiscretizer{
 public:
  UniformGrid(vec & low,  // Least vertex of rectangular region
	      vec & high, // Greatest vertex of rectangular region
	      uvec & num_cells, // Resolution (in cells) of discretization
	      uint special_nodes); // Number of special (i.e. oob) nodes 

  TypedPoints get_spatial_nodes() const;
  TypedPoints get_cell_centers() const;
  umat get_cell_node_indices() const;

  uint max_spatial_node_index() const;
  uint max_node_index() const;

  uvec cell_coords_to_low_node_indices(const Coords & coords) const;
  umat cell_coords_to_vertex_indices(const Coords & coords) const;

  Coords points_to_cell_coords(const TypedPoints & points) const;
  TypedPoints cell_coords_to_low_points(const Coords & coords) const;
  
  mat points_to_cell_nodes_dist(const TypedPoints & points,
				    const Coords & coords) const;
  
  ElementDist points_to_element_dist(const TypedPoints &) const;
  template <typename T> T base_interpolate(const Points & points,
					   const T& data) const;
  vec interpolate(const Points & points,
		  const vec & values) const;
  mat interpolate(const Points & points,
		  const mat & values) const;
  mat find_bounding_box() const;

  mat cell_gradient(const vec & value) const;


  // private: 
  vec m_low;
  vec m_high;
  uvec m_num_cells;
  uvec m_num_nodes;
  vec m_width;
  
  uint n_special_nodes;
  uint n_dim;
};
ElementDist build_sparse_dist(uint n_nodes, umat vert_indices, mat weights);

#endif
