#include <iostream>
#include <assert.h>
#include "grid.h"
#include "misc.h"

uvec c_order_stride(const uvec & points_per_dim){
  /* 
     Coeffs to converts from grid coords to indicies
     This logic is used in laying out matrices in memory and also called 
     a "stride"
     E.g. if the mesh is 3x2 (nodes), then the (node) indices are:

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

uvec c_order_cell_shift(const uvec & points_per_dim){
  /* 
     Consider a D-dimenional cell of the grid. There is a least indexed 
     vertex according to C order. The pattern of indices relative to this 
     least vertex is fixed: different cells are just offset by a different
     minimum vertex.

     We index these vertices by their coordinate in binary. E.g. in 2D:

     01 - 11     1 - 3
      |   |   => |   |
     00 - 10     0 - 2

     This function returns those constant index offsets.

     E.g. if the grid is:

     6 - 7 - 8
     |   |   |
     3 - 4 - 5
     |   |   |
     0 - 1 - 2

     Notices that the cell <0,3,1,4> =  <4,7,5,8> - 4; it's the same pattern.

     points_per_dim: the number of points in the grid along each
     dimension.

     TODO: better comment explaining this
  */
  
  uint D = points_per_dim.n_elem;
  uvec strides = c_order_stride(points_per_dim);
  uint V = pow(2,D);
  
  uvec shifts = zeros<uvec>(V);
  uvec idx;
  for(uint d = 0; d < D; d++){
    idx = find(binmask(d,D));
    shifts(idx) += strides(d);
  }
  return shifts;
}

Indices coords_to_indices(const Coords & coords,
			  const uvec & num_entity){
  /*
   * Takes in a coordinate block
   */
  uint N = coords.num_total;
  Indices idx = Indices(N);

  // Out of bound indices
  uint num_spatial_index = prod(num_entity);
  idx(coords.oob.indices) = num_spatial_index + coords.oob.type;

  // In bound indices
  // Note that for 2-3 dim this functionality is provided by sub2ind
  uvec stride = c_order_stride(num_entity);
  Indices inbound_idx = coords.coords * stride;
  idx(coords.indices) = inbound_idx;
  
  // Check if indices are smaller than 3D
  if(coords.dim == 1){
    assert(all(inbound_idx == coords.coords.col(0)));
  }
  if(coords.dim == 2){
    // Kludge. Not sure how to convert from uvec
    // to return type of 'size' (SizeMat). conv_to doesn't work.
    Indices check_idx = sub2ind(size(num_entity(0),
				     num_entity(1)),
				coords.coords.t());
    assert(all(inbound_idx == check_idx));
  }
  if(coords.dim == 3){
    Indices check_idx = sub2ind(size(num_entity(0),
				     num_entity(1),
				     num_entity(2)),
				coords.coords.t());
    assert(all(inbound_idx == check_idx));
  }
  return idx;
}

UniformGrid::UniformGrid(vec & low,
			  vec & high,
			 uvec & num_cells) :
  m_low(low),
  m_high(high),
  m_num_cells(num_cells),
  m_num_nodes(num_cells + 1),
  m_width((high - low) / num_cells),
  m_dim(low.n_elem){
  uint D = low.n_elem;
  assert(D == high.n_elem);
  assert(D == num_cells);
}


TypedPoints UniformGrid::get_spatial_nodes(){
  /*
   * Might want to make this a more general function.
   * Generates grid points in order.
   * Later dimensions cycle faster, e.g.:
   * [0 0]
   * [0 1]
   * [0 2]
   * [1 0]
   *  ...
   */
  uint D = m_low.n_elem;
  uint N = number_of_spatial_nodes();
  Points points = Points(N,d);

  // Which cycle iteration we are on
  uint cycle = 1;
  // How many times to repeat a value before moving to the next
  uint repeat = prod(m_num_nodes.tail(D-1));
  
  for(uint d = 0; d < D; d++){
    uint idx = 0;
    uint nn = m_num_nodes(d);
    vec values = linspace<vec>(m_low(d), m_high(d), nn);

    // Sanity checks (also explains the update)
    assert(cycle == prod(m_num_nodes.head(d)));
    assert(repeat == prod(m_num_nodes.tail(D - d - 1)));
    assert(N == cycle * repeat * nn);
      
    for(uint c = 0; c < cycle; c++){
      for(uint value_idx = 0; value_idx < nn; value_idx++){
	for(uint r = 0; r < repeat; r++){
	  points(idx,d) = values(value_idx);
      }
      }
    }

    // Shift over the cycle and repeat
    cycle *= m_num_nodes(d);
    if(d < D - 1){
      assert(repeat > 1);
      assert(0 == mod(repeat, m_num_nodes(d+1)));
      repeat /= m_num_nodes(d+1);
    }    
  }
}


  virtual TypedPoints get_cell_centers() const = 0;
  virtual umat get_cell_node_indices() const = 0;


/*
 * OLD CODE REGION
 */

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
  return coords_to_indices(coords,m_num_cells);
}
Indices UniformGrid::cell_coords_to_low_node_indices(const Coords & coords){
  return coords_to_indices(coords,m_num_nodes);
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


VertexIndices UniformGrid::cell_coords_to_vertices(const Coords & coords){
  uint N = coords.num_total;
  uint D = coords.dim;
  uint V = pow(2,D);
  
  uvec shift = c_order_cell_shift(m_num_nodes);
  assert(shift.n_elem == V);
  
  Indices low_idx = cell_coords_to_low_node_indices(coords);
  VertexIndices vertices = VertexIndices(N,V);
  uvec col_idx;
  uvec inb_idx = coords.indices;
  uvec oob_idx = coords.oob.indices;
  for(uint v = 0; v < V; v++){
    col_idx = uvec({v});
    vertices(inb_idx,col_idx) = low_idx(inb_idx) + shift(v);
    vertices(oob_idx,col_idx) = low_idx(oob_idx);
  }
  return vertices;
}

RelDist UniformGrid::points_to_low_node_rel_dist(const Points & points,
						 const Coords & coords){
  uint N = coords.num_total;
  uint D = coords.dim;

  Points low_node = cell_coords_to_low_node(coords);
  RelDist dist = RelDist(N,D);
  uvec inb_idx = coords.indices;
  uvec oob_idx = coords.oob.indices;
  mat diff = points.rows(inb_idx) - low_node.rows(inb_idx);
  rowvec div = conv_to<rowvec>::from(m_width);
  dist.rows(inb_idx) = row_divide(diff,div);
  dist.rows(oob_idx).fill(0);
  return dist;
}

ElementDist UniformGrid::points_to_element_dist(const Points & points){
  
  Coords coords = points_to_cell_coords(points);
  VertexIndices vertices = cell_coords_to_vertices(coords);
  RelDist rel_dist =  points_to_low_node_rel_dist(points,coords);

  uint N = points.n_rows;
  uint D = points.n_cols;
  uint IN = coords.num_inbound;
  uint NN = prod(m_num_nodes) + 2*D;
  uint V = pow(2,D);
  uint halfV = pow(2,D-1);

  // Calculate the interpolation weights for each point
  mat weights = ones<mat>(IN,V);
  bvec mask;
  uvec inb_idx = coords.indices;
  uvec oob_idx = coords.oob.indices;
  uvec col_idx;
  mat rep_dist;
  for(uint d = 0; d < D; d++){
    // Mask for whether the dth bit is on in the
    // binary rep of the bth position
    mask = binmask(d,D);
    col_idx = uvec({d});

    rep_dist = repmat(rel_dist(inb_idx,col_idx),
		      1, halfV);
    
    weights.cols(find(mask > 1e-12)) %= rep_dist;		       
    weights.cols(find(mask <= 1e-12)) %= 1- rep_dist;
  }

  // Check the weights of the low and high nodes.
  // position 0 -> 00...0 is the low node, so weight should be prod(rel_dist)
  // position V-1 -> 11...1, so weight should be prod(1-rel_dist)
  vec low_node_weights = prod(1 - rel_dist.rows(inb_idx),1);
  vec hi_node_weights = prod(rel_dist.rows(inb_idx),1);
  assert(approx_equal(weights.col(0),low_node_weights,"absdiff",1e-12));
  assert(approx_equal(weights.col(V-1),hi_node_weights,"absdiff",1e-12));

  for(uint d = 0; d < D; d++){
    weights(find(weights.col(d) <= ALMOST_ZERO),
	    uvec({d})).fill(0);
  }

  VertexIndices inbound_vertices = vertices.rows(coords.indices);
  Indices oob_vertices = vertices(coords.oob.indices,uvec({0}));
  ElementDist distrib = pack_vertices_and_weights(NN,
						  coords.indices,
						  inbound_vertices,
						  weights,
						  coords.oob.indices,
						  oob_vertices);     

  return distrib; 
}

ElementDist pack_vertices_and_weights(uint num_total_nodes,
				      Indices inbound_indices,
				      VertexIndices inbound_vertices,
				      mat inbound_weights,
				      Indices oob_indices,
				      Indices oob_vertices){
  assert(inbound_vertices.n_rows == inbound_indices.n_elem);
  assert(oob_indices.n_elem == oob_vertices.n_elem);
  assert(size(inbound_vertices) == size(inbound_weights));
  
  uint N = num_total_nodes;
  uint M = inbound_vertices.n_rows + oob_vertices.n_elem;
  uint INB_N = inbound_vertices.n_elem;
  uint OOB_N = oob_vertices.n_elem;
  uint NNZ = INB_N + OOB_N;
 

  // Set the location
  umat loc = umat(2,NNZ);
  // Flatten vertices and indices into rows
  uvec flat_inb_vertices = vectorise(inbound_vertices);
  umat rep_inb_idx = repmat(inbound_indices,
			    1, inbound_vertices.n_cols);
  uvec flat_inb_idx = vectorise(rep_inb_idx);
  // Check sizes
  assert(flat_inb_vertices.n_elem == INB_N);
  assert(flat_inb_idx.n_elem == INB_N);
  loc(0,span(0,INB_N-1)) = flat_inb_vertices.t();
  loc(1,span(0,INB_N-1)) = flat_inb_idx.t();

  cout << "LOC:\n" << loc;

  // Out of bound location
  loc(0,span(INB_N,NNZ-1)) = oob_vertices.t();
  loc(1,span(INB_N,NNZ-1)) = oob_indices.t();

  // Set the data
  vec data = vec(INB_N + OOB_N);
  data(span(0,INB_N-1)) = vectorise(inbound_weights); // Inbound
  data(span(INB_N,NNZ-1)).fill(1); // Out of bounds

  return sp_mat(loc,data,N,M);  
}
				      
