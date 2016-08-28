#include <iostream>
#include <assert.h>
#include "grid.h"
#include "misc.h"

ostream& operator<< (ostream& os, const OutOfBounds& oob){
  os << "Indices: " << oob.indices.t();
  os << "Type: " << oob.type.t();
  return os;
}

ostream& operator<< (ostream& os, const Coords& coords){
  os << "Indices: " << coords.indices.t();
  os << "Coords:\n" << coords.coords;
  return os;
}


UniformGrid::UniformGrid(vec & low,
			  vec & high,
			 uvec & num_cells) :
  m_low(low),
  m_high(high),
  m_num_cells(num_cells),
  m_num_nodes(num_cells+1),
  m_width((high - low) / num_cells),
  m_dim(low.n_elem){

  std::cout << m_width << std::endl;
}

OutOfBounds UniformGrid::points_to_out_of_bounds(const Points & points){
  // Two passes: identify the OOB points, then classify
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(D == m_dim);

  // Generate the mask
  OutOfBounds oob;
  vec fuzzed = m_high + GRID_FUZZ;
  bvec normal = in_intervals(points,
			     m_low,
			     fuzzed);
  oob.mask = lnot(normal);
  
  // Find and count the oob indices
  oob.indices = find(oob.mask);
  oob.num = oob.indices.n_elem;

  // Classify the oob points
  Points oob_points = points.rows(oob.indices);
  oob.type = vec(oob.num);
  for(int d = D-1; d >= 0; d--){
    // Lower dimensional violations overwrites higher dimensional ones.
    oob.type(find(oob_points.col(d) < m_low(d))).fill(2*d);
    oob.type(find(oob_points.col(d) > m_high(d) + GRID_FUZZ)).fill(2*d+1);
  }

  return oob;
}


Coords UniformGrid::points_to_cell_coords(const Points & points){
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(D == m_dim);

  Coords coords;
  coords.oob = this->points_to_out_of_bounds(points);

  coords.indices = find(coords.oob.mask == 0);
  coords.num = coords.indices.n_elem;
  assert(coords.num == N - coords.oob.num);
  
  if(coords.indices.n_elem == 0) return coords;

  Points inbound = points.rows(coords.indices);
  mat diff = row_diff(inbound,m_low.t());
  mat scaled = row_divide(diff, conv_to<rowvec>::from(m_width));
  coords.coords = conv_to<umat>::from(floor(scaled));

  // 
  for(uint d = 0; d < D; d++){
    uint n = m_num_cells(d);
    uvec row_idx = find(coords.coords.col(d) == n);
    uvec col_idx = {d};
    coords.coords.submat(row_idx,col_idx).fill(n-1);
  }
  return coords;
}
