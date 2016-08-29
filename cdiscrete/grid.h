#ifndef __Z_GRID_INCLUDED__
#define __Z_GRID_INCLUDED__

#include <armadillo>
#include <climits>
#include <map>
#include "misc.h"

using namespace std;
using namespace arma;

#define GRID_FUZZ 1e-15

typedef mat Points;
typedef uvec Indices;
typedef umat VertexIndices;
typedef sp_mat ElementDist;
typedef mat RelDist;

struct OutOfBounds{
  // Contains out of bounds information
  bvec mask;
  uvec indices;
  uvec type; // +d / -d for violate in dth dim
  uint num;
};
ostream& operator<< (ostream& os, const OutOfBounds& oob);


struct Coords{
  // Contains coordinate information.
  // This is the useful internal structure
  OutOfBounds oob;  
  uvec indices;
  umat coords;
  
  uint num_inbound;
  uint num_total;
  uint dim;
};
ostream& operator<< (ostream& os, const Coords& oob);

uvec c_order_stride(const uvec & points_per_dim);
uvec c_order_cell_shift(const uvec & points_per_dim);
Indices coords_to_indices(const Coords & coords,
			  const uvec & num_entity);
class UniformGrid{
 public:
  UniformGrid(vec & low,
	      vec & high,
	      uvec & num_cells);

  OutOfBounds points_to_out_of_bounds(const Points &);
  Coords points_to_cell_coords(const Points &);
  Indices cell_coords_to_cell_indices(const Coords &);
  Indices cell_coords_to_low_node_indices(const Coords & coords);
  Points cell_coords_to_low_node(const Coords &);  
  VertexIndices cell_coords_to_vertices(const Coords &);
  RelDist points_to_low_node_rel_dist(const Points &,
				      const Coords &);
  ElementDist points_to_element_dist(const Points &);

  // private: 
  vec m_low;
  vec m_high;
  uvec m_num_cells;
  uvec m_num_nodes;
  vec m_width;

  uint m_dim;
};

#endif
