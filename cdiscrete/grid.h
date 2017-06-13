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
Indices coords_to_indices(const Coords & coords,
			  const uvec & num_entity);
class UniformGrid : public TypedDiscretizer{
 public:
  UniformGrid(vec & low,
	      vec & high,
	      uvec & num_cells);

 public:
  virtual TypedPoints get_spatial_nodes() const = 0;
  virtual TypedPoints get_cell_centers() const = 0;
  virtual umat get_cell_node_indices() const = 0;

  virtual uint number_of_all_nodes() const = 0;
  virtual uint number_of_spatial_nodes() const = 0;
  virtual uint number_of_cells() const = 0;
  
  virtual ElementDist points_to_element_dist(const TypedPoints &) const = 0;
  virtual vec interpolate(const Points & points,
                          const vec & values) const = 0;
  virtual mat interpolate(const Points & points,
                          const mat & values) const = 0;
  virtual mat find_bounding_box() const = 0;

  virtual mat cell_gradient(const vec & value) const = 0;


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
