#ifndef __Z_GRID_INCLUDED__
#define __Z_GRID_INCLUDED__

#include <armadillo>
#include <climits>
#include <climits>
#include <map>

#include "discretizer.h"
#include "misc.h"
#include "points.h"

arma::uvec c_order_stride(const arma::uvec & points_per_dim);
arma::uvec c_order_cell_shift(const arma::uvec & points_per_dim);

class Coords;  // Forward declare
Coords indices_to_coords(const arma::uvec & grid_size,
			 const arma::uvec & indices);
arma::uvec coords_to_indices(const arma::uvec & grid_size,
		       const Coords & coords);

class Coords{
 public:
  static const uint DEFAULT_OOB_TYPE = 1;
  static const uint SPATIAL_TYPE = 0;
  static const arma::sword SPECIAL_FILL = LONG_MIN;

  Coords(const arma::imat & coords);
  Coords(const arma::imat & coords, const TypeRegistry & reg);
  Coords(const TypedPoints & points);

  bool check() const;
  bool check(const arma::uvec & grid_size) const;
  bool _coord_check(const arma::uvec & grid_size) const;

  void _mark(const arma::uvec & indices, uint coord_type);
  void _mark(const TypeRegistry & reg);
  TypeRegistry _find_oob(const arma::uvec & grid_size,
			 uint type = Coords::DEFAULT_OOB_TYPE) const;

  uint num_coords() const;
  uint num_spatial() const;
  uint num_special() const;
  uint max_spatial_index(const arma::uvec & grid_size) const;

  bool is_special(uint idx) const;
  arma::uvec get_spatial() const;
  arma::uvec get_special() const;

  arma::uvec get_indices(const arma::uvec & grid_size) const;

  bool equals(const Coords & other) const;

  friend std::ostream& operator<<(std::ostream& os, const Coords& c);  
  
  arma::imat m_coords;
  uint n_rows;
  uint n_dim;
  TypeRegistry m_reg;
};

class UniformGrid : public TypedDiscretizer{
 public:
  UniformGrid(arma::vec & low,  // Least vertex of rectangular region
	      arma::vec & high, // Greatest vertex of rectangular region
	      arma::uvec & num_cells, // Resolution (in cells) of discretization
	      uint special_nodes); // Number of special (i.e. oob) nodes 

  TypedPoints get_spatial_nodes() const;
  TypedPoints get_cell_centers() const;
  arma::umat get_cell_node_indices() const;

  uint number_of_all_nodes() const;
  uint number_of_spatial_nodes() const;
  uint number_of_cells() const;

  uint max_spatial_node_index() const;
  uint max_node_index() const;

  arma::uvec cell_coords_to_low_node_indices(const Coords & coords) const;
  arma::umat cell_coords_to_vertex_indices(const Coords & coords) const;

  Coords points_to_cell_coords(const TypedPoints & points) const;
  TypedPoints cell_coords_to_low_points(const Coords & coords) const;

  TypedPoints apply_rules_and_remaps(const TypedPoints & points) const;
  
  arma::mat points_to_cell_nodes_dist(const TypedPoints & points) const;
  arma::mat points_to_cell_nodes_dist(const TypedPoints & points,
				      const Coords & coords) const;
  
  ElementDist points_to_element_dist(const TypedPoints &) const;
  template <typename T> T base_interpolate(const TypedPoints & points,
					   const T& data) const;
  arma::vec interpolate(const TypedPoints & points,
			const arma::vec & values) const;
  arma::mat interpolate(const TypedPoints & points,
			const arma::mat & values) const;
  arma::mat find_bounding_box() const;

  arma::mat cell_gradient(const arma::vec & value) const;


  // private: 
  arma::vec m_low;
  arma::vec m_high;
  arma::uvec m_num_cells;
  arma::uvec m_num_nodes;
  arma::vec m_width;

  TypeRuleList m_rule_list;
  NodeRemapperList m_remap_list;
  
  uint n_special_nodes;
  uint n_dim;
};
ElementDist build_sparse_dist(uint n_nodes, umat vert_indices, arma::mat weights);

#endif
