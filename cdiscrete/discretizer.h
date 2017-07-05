#ifndef __Z_DISCRETE_INCLUDED__
#define __Z_DISCRETE_INCLUDED__

#include <armadillo>
#include "points.h"

using namespace std;
using namespace arma;

typedef uvec   Indices;
typedef umat   VertexIndices;
typedef sp_mat ElementDist;

class Discretizer{
 public:
  virtual Points get_spatial_nodes() const = 0;
  virtual Points get_cell_centers() const = 0;
  virtual umat get_cell_node_indices() const = 0;

  virtual uint number_of_all_nodes() const = 0;
  virtual uint number_of_spatial_nodes() const = 0;
  virtual uint number_of_cells() const = 0;
  
  virtual ElementDist points_to_element_dist(const Points &) const = 0;
  virtual vec interpolate(const Points & points,
                          const vec & values) const = 0;
  virtual mat interpolate(const Points & points,
                          const mat & values) const = 0;
  virtual mat find_bounding_box() const = 0;

  virtual mat cell_gradient(const vec & value) const = 0;
};

class TypedDiscretizer{
 public:
  virtual TypedPoints get_spatial_nodes() const = 0;
  virtual TypedPoints get_cell_centers() const = 0;
  virtual umat get_cell_node_indices() const = 0;

  virtual uint number_of_all_nodes() const = 0;
  virtual uint number_of_spatial_nodes() const = 0;
  virtual uint number_of_cells() const = 0;
  
  virtual ElementDist points_to_element_dist(const TypedPoints &) const = 0;
  virtual vec interpolate(const TypedPoints & points,
                          const vec & values) const = 0;
  virtual mat interpolate(const TypedPoints & points,
                          const mat & values) const = 0;
  virtual mat find_bounding_box() const = 0;

  virtual mat cell_gradient(const vec & value) const = 0;
};

struct BaryCoord{
  /*
    Holds the barycentric coordinates. These are the unique weights describing
    a point as a convex combination of vertices in the enclosed face.
    Also indicates if the 
  */
  BaryCoord();
  BaryCoord(bool,const uvec&,const vec&);
  bool oob;
  uvec indices;
  vec  weights;
};
ostream& operator<< (ostream& os, const BaryCoord& coord);

#endif
