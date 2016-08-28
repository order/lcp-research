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

uvec c_order_stride(const uvec & points_per_dim){
  /* Coeffs to converts from point grid coords to node indicies
   These are also called "strides"
   E.g. if the mesh is 3x2, then the indices are:

   4 - 5
   |   |
   2 - 3
   |   |
   0 - 1

   So, to figure out the index of grid coord (x,y), we just multiply
   it by the coefficients [2,1].
   
   For 2 and 3 dim, we should exactly match the output of sub2ind
  */
  
  uint D = points_per_dim.n_elem;
  uvec stride = uvec(D);

  uint agg = 1;
  for(uint i = D; i > 0; --i){
    stride(i-1) = agg;
    agg *= points_per_dim(i-1);
  }
  return stride;
}



UniformGrid::UniformGrid(vec & low,
			  vec & high,
			 uvec & num_cells) :
  m_low(low),
  m_high(high),
  m_num_cells(num_cells),
  m_num_nodes(num_cells+1),
  m_width((high - low) / num_cells),
  m_dim(low.n_elem){}

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
  oob.type = uvec(oob.num);
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
  coords.num_total = N;
  coords.num_inbound = coords.indices.n_elem;
  assert(coords.num_total == coords.num_inbound + coords.oob.num);
  coords.dim = D;
  
  if(coords.indices.n_elem == 0) return coords;

  Points inbound = points.rows(coords.indices);
  mat diff = row_diff(inbound,conv_to<rowvec>::from(m_low));
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

Indices UniformGrid::cell_coords_to_cell_indices(const Coords & coords){
  uint N = coords.num_total;
  Indices idx = Indices(N);

  // Out of bound indices
  uint num_spatial_index = prod(m_num_cells);
  idx(coords.oob.indices) = num_spatial_index + coords.oob.type;

  // In bound indices
  // Note that for 2-3 dim this functionality is provided by sub2ind
  uvec stride = c_order_stride(m_num_cells);
  Indices inbound_idx = coords.coords * stride;
  idx(coords.indices) = inbound_idx;

  // Check if indices are smaller than 3D
  if(coords.dim == 1){
    assert(all(inbound_idx == coords.coords.col(0)));
  }
  if(coords.dim == 2){
    // Kludge. Not sure how to convert from uvec
    // to return type of 'size' (SizeMat). conv_to doesn't work.
    Indices check_idx = sub2ind(size(m_num_cells(0),m_num_cells(1)),
				coords.coords.t());
    assert(all(inbound_idx == check_idx));
  }
  if(coords.dim == 3){
    Indices check_idx = sub2ind(size(m_num_cells(0),
				     m_num_cells(1),
				     m_num_cells(2)),
				coords.coords.t());
    assert(all(inbound_idx == check_idx));
  }
  return idx;
}

Points UniformGrid::cell_coords_to_low_node(const Coords & coords){
  Points low_points = Points(coords.num_total,coords.dim);
  low_points.rows(coords.oob.indices).fill(datum::nan);
  mat scaled = row_mult(conv_to<mat>::from(coords.coords),
			conv_to<rowvec>::from(m_width));
  low_points.rows(coords.indices) = row_add(scaled,
					    conv_to<rowvec>::from(m_low));
  return low_points;
}
