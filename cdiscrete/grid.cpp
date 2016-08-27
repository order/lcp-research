#include <iostream>
#include <assert.h>
#include "grid.h"
#include "misc.h"

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
  oob.type = ivec(oob.num);
  for(int d = D-1; d >= 0; d--){
    // Lower dimensional violations overwrites higher dimensional ones.
    oob.type(oob_points.col(d) < m_low(d)).fill(-d);
    oob.type(oob_points.col(d) > m_high(d) + GRID_FUZZ).fill(d);
  }

  return oob;
}
