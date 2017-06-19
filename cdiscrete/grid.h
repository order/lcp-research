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

typedef std::map<uint,uint> CoordTypeRegistry;

uvec c_order_stride(const uvec & points_per_dim);
uvec c_order_cell_shift(const uvec & points_per_dim);
Indices coords_to_indices(const Coords & coords,
			  const uvec & num_entity);

class Coords{
 public:
  static const uint SPATIAL_COORD = 0;
  static const uint OOB_COORD = 1;
  
  Coords(const imat & coords);

  static bool _coord_check() const;
  static Coords _indicies_to_coords(const uvec & grid_size,
				  const uvec & indices);
  static uvec _coords_to_indices(const uvec & grid_size,
				 const Coords & coords);
  void _mark(const uvec & indices, uint coord_type);
  void _mark(const TypeRegistry & reg);
 
  void restrict(const uvec & grid_size);

  uint number_of_spatial_coords() const;
  uint number_of_all_coords() const;
  uint number_of_special_coords() const;

  uvec get_indices() const;
  
  imat m_coords;
  uint n_rows;
  uint n_dim;
  CoordTypeRegistry m_type_map;
}

class UniformGrid : public TypedDiscretizer{
 public:
  UniformGrid(vec & low,  // Least vertex of rectangular region
	      vec & high, // Greatest vertex of rectangular region
	      uvec & num_cells); // Fineness (in cells) of discretization

  TypedPoints get_spatial_nodes() const = 0;
  TypedPoints get_cell_centers() const = 0;
  umat get_cell_node_indices() const = 0;

  uint number_of_all_nodes() const = 0;
  uint number_of_spatial_nodes() const = 0;
  uint number_of_cells() const = 0;
  
  ElementDist points_to_element_dist(const TypedPoints &) const = 0;
  vec interpolate(const Points & points,
		  const vec & values) const = 0;
  mat interpolate(const Points & points,
		  const mat & values) const = 0;
  mat find_bounding_box() const = 0;

  mat cell_gradient(const vec & value) const = 0;


  // private: 
  vec m_low;
  vec m_high;
  uvec m_num_cells;
  uvec m_num_nodes;
  vec m_width;

  uint m_dim;
};

ElementDist pack_vertices_and_weights(uint num_total_nodes,
				      Indices inbound_indices,
				      VertexIndices inbound_vertices,
				      mat inbound_weights,
				      Indices oob_indices,
				      Indices oob_vertices);

#endif
