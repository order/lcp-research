#include "kuhn_grid.h"

using namespace std;
using namespace arma;


mat box2points(const mat & bbox){
  /*
   * Generate the vertex points of the boundary box
   */
  
  uint D = bbox.n_rows;
  vector<vec> grids;
  for(auto const & it : bbox.row_iterator()){
    grids.push_back(conv_to<vec>::from(it));
  }
  
  mat points = make_points<mat,vec>(grids);
  assert(size(2**D, D) == size(points));
  
  return points
    }

mat slice2points(const mat & bbox, uint dim){
  /*
   * Generate the vertex points of the boundary box
   */
  
  uint D = bbox.n_rows;
  double mid_pt = (bbox(dim, 0) + bbox(dim,1)) / 2;
  
  vector<vec> grids;
  for(uint d = 0; d < D, d++){
    if(dim == d){
      grids.push_back(mid_pt * ones<vec>(1));
    }
    else{
      grids.push_back(conv_to<vec>::from(it));
    }
  }
  
  mat points = make_points<mat,vec>(grids);
  assert(size(2**D, D) == size(points));
  
  return points;
}


KuhnGrid::KuhnGrid(const mat & bbox) : _bbox(bbox){
  /*
   * Create a Kuhn grid with a single cell
   */
  mat points = box2points(bbox); // Generate the points
  _append_to_points(points);
  uint N = _vert_points.size();

  // One cell, all points in it
  _cell_map[0] = regspace<uvec>(0, N-1);
  _cell_bbox_map[0] = bbox;

  // All points in that cell
  for(uint i = 0; i < N; i++){
    _point_map[i] = zeros<uvec>(1);
  }
}

void _append_to_points(const mat & points){
  for(auto const & it : points.row_iterator()){
    _vert_points.push_back(conv_to<vec>::from(it));
  }
}
 
void split(uint cell_id, uint dim){
  uint cell_bbox = _cell_bbox_map[cell_id];
  uint D = bbox.n_rows;
  uint N = _vert_points.size();

  // Make the new points
  mat new_points = slice2points(cell_bbox, dim);
  _append_to_points(new_points);

  // Generate the new ids
  uint n = new_points.n_rows;
  assert(n == 2**(D-1));  
  uvec new_ids = regspace(N, N+n-1);
  assert(n == new_ids.n_elems);
  _append_to_points(new_points);

  //
}
