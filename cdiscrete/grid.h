#ifndef __Z_GRID_INCLUDED__
#define __Z_GRID_INCLUDED__

#include <armadillo>
#include <climits>
#include <map>
#include "misc.h"

using namespace std;
using namespace arma;

#define GRID_FUDGE 1e-15

typedef mat Points;
typedef uvec Indices;
typedef mat VertexIndicies;
typedef sp_mat ElementDist;
typedef mat RelDist;

struct OutOfBounds{
  // Contains out of bounds information
  bvec oob_mask;
  uvec oob_idx;
  ivec oob_type; // +d / -d for violate in dth dim
};

struct Coords{
  // Contains coordinate information.
  // This is the useful internal structure
  OutOfBounds oob;  
  uvec inbound_idx;
  umat coords;
};

class UniformGrid{
 public:
  UniformGrid(vec & low,
	      vec & high,
	      uvec & num_cells);

  ElementDist points_to_element_dist(const Points &);
  Coords points_to_cell_coords(const Points &);
  Indices cell_coords_to_cell_indicies(const Coords &);
  Indices cell_coords_to_low_node(const Coords &);
  VertexIndicies cell_coords_to_vertices(const Coords &);
  RelDist cell_coords_to_low_node_rel_dist(const Coords &);  

 private: 
  vec m_low;
  vec m_high;
  uvec m_num_cells;
  uvec m_num_nodes;
  vec m_width;
};

#endif
