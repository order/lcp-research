#ifndef __Z_KUHN_INCLUDED__
#define __Z_KUHN_INCLUDED__

#include <armadillo>
#include <climits>
#include <climits>
#include <map>

#include "discretizer.h"
#include "misc.h"
#include "points.h"

typedef std::map<uint, arma::uvec> Cell2PointMap;
typedef std::map<uint, arma::mat> Cell2BBoxMap;
typedef std::map<uint, arma::uvec> Point2CellMap;

typedef vector<vec> StdPoints;

class KuhnGrid : public TypedDiscretizer{
 public:
  KuhnGrid(const arma::mat & bbox);

  void split(uint cell_id, uint dim);

  TypedPoints get_all_nodes() const;
  TypedPoints get_spatial_nodes() const;
  TypedPoints get_cell_centers() const;
  umat get_cell_node_indices() const;

  uint number_of_all_nodes() const;
  uint number_of_spatial_nodes() const;
  uint number_of_cells() const;
  
  ElementDist points_to_element_dist(const TypedPoints &) const;
  vec interpolate(const TypedPoints & points,
                          const vec & values) const;
  mat interpolate(const TypedPoints & points,
                          const mat & values) const;
  mat find_bounding_box() const;

  mat cell_gradient(const vec & value) const;

  mat _bbox;  // Overall boundary of rectangular space
  StdPoints _vert_points; // Positions of the vertex points.
  Cell2PointMap _cell_map; // Which points are in each cell
  Cell2BBoxMap _cell_bbox_map; // Boundary of each cell
  Point2CellMap _point_map; // Which cells are each point in.
};

#endif
