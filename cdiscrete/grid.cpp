#include <iostream>
#include <assert.h>
#include "grid.h"

using namespace std;
using namespace arma;


/**************************************************************************
 * STRIDE HELPER FUNCTIONS *
 ***************************/




uvec c_order_stride(const uvec & grid_size){
  /* 
     Coeffs to converts from grid coords to indicies
     This is a C-style layout, which is row-major for 2d.
     Elements of a row will be adjacent in memory, elements of a column
     may be disjoint.
     E.g. if the mesh is 3x2 (nodes), then the (node) indices are:

     4 - 5
     |   |
     2 - 3
     |   |
     0 - 1

     To figure out the index of grid coord (x,y), we just multiply
     it by the coefficients [2,1]; this vector is what the function returns.

     For higher order tensors, the indexing will be contiguous for the last 
     dimension.
   
     NB: For 2 and 3 dim, we should exactly match the output of arma::sub2ind.
  */
  
  uint D = grid_size.n_elem;
  uvec stride = uvec(D);

  uint agg = 1;
  for(uint d = D; d > 0; --d){
    stride(d-1) = agg;
    agg *= grid_size(d - 1);
  }

  // Check for 2 and 3 dimensions
  if(2 == D){
    SizeMat sz = uvec2sizemat(grid_size);
    umat I = eye<umat>(2,2);
    uvec stride_check = sub2ind(sz, I);
    assert(all(stride_check == stride));
  }
  else if(3 == D){
    SizeCube sz = uvec2sizecube(grid_size);
    umat I = eye<umat>(3,3);
    uvec stride_check = sub2ind(sz, I);     
    assert(all(stride_check == stride));
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
     This patter, <0,3,1,4> is the "C-order cell shift".

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


/************************************************************************
 * COORDINATE CLASS *
 ********************/


Coords::Coords(const imat & coords){
  m_coords = coords;
  n_rows = coords.n_rows;
  n_dim = coords.n_cols;

  uvec special = find_nonfinite(coords.col(0));
  _mark(special, Coords::OOB_TYPE);
}


bool Coords::_check(const uvec & grid_size) const{
  assert(_coord_check(grid_size, m_coords));
  assert(_type_reg_check(m_type_map, m_coords));
}


bool Coords::_coord_check(const uvec & grid_size, const umat & coords) const{
  /*
   * Check to make sure that the coordinates make sense with the supplied
   * grid sizes, and that non-spatial coords are NaN'd out correctly
   */

  uvec ref_nan = find_nonfinite(coords.col(0));
  uvec ref_fin = find_finite(coords.col(0));
  assert(0 == find_finite(coords.rows(ref_nan)));
  assert(0 == find_non_finite(coords.rows(ref_fin)));
  
  imat signed_coords = conv_to<imat>::from(coords);
  signed_coords.each_row() -= conv_to<ivec>::from(grid_size);
  assert(all(signed_coords < 0));
  return true;
}

bool Coords::_type_reg_check(const TypeRegistry & registry, const umat & coords){
  uvec find_res = find_nonfinite(coords.col(0));
  for(auto const& it : find_res){
    assert(registry.end() != registry.find(*it));
  }
  return true;
}



umat Coords::_indicies_to_coords(const uvec & grid_size,
				 const uvec & indices) const {
  /*
   * Convert indices into coordinates by repeated modulus.
   * NB: makes sense because c-order stride is in decreasing order
   */

  uint max_idx = prod(grid_size);
  assert(all(indices <= max_idx)); // Only one OOB
  
  uvec stride = c_order_stride(grid_size);
  umat coords = umat(indices.n_elem, n_dim);

  umat mod_res = umat(indices.n_elem, 2);
  mod_res.col(1) = indices;
  for(uint d = 0; d < n_dim; d++){
    mod_res = divmod(mod_res.col(1), stride(d));
    coords.col(d) = mod_res.col(0);
  }

  // NaN out any non-spatial nodes
  // NB: assumption is that there are relatively few of these usually.
  uvec non_spatial = find(indices >= max_idx);
  coords.rows(non_spatial).fill(datum::nan);

  assert(_coord_check(grid_size, coords));
  
  return coords;
}


uvec Coords::_coords_to_indices(const uvec & grid_size,
				const Coords & coords) const{
  // Calculate indices for normal coords

  // TODO: convert coords to uvec somehow
  uvec stride = c_order_stride(grid_size);
  umat indices = coords.m_coords * stride; 

  // OOB special indices
  TypeRegistry oob_reg = coords._find_oob(grid_size);
  uint special_idx = max_spatial_index(grid_size);
  for(auto const & it : oob_reg){
    assert(SPATIAL_TYPE != it->second);
    indices(it.first) = it.second + special_idx;
  }

  // Check that everything has been updated.
  assert(0 == find_nonfinite(indices));
  
  return indices;
}

void Coords::_mark(const uvec & indices, uint coord_type){
  /*
   * Go through and add the indices to the type map with the supplied
   * coord_type
   */
  assert(coord_type != SPATIAL_COORD);
  for(auto const & it : indices){
    m_type_map[*it] = coord_type;
    m_coords.row(*it).fill(datum::nan);
  }
}


void Coords::_mark(const TypeRegistry & reg){
  /*
   * Go through and add the indices to the type map with the supplied
   * coord_type
   */
  for(auto const & it : indices){
    assert(it->second != SPATIAL_TYPE);
    m_type_map[it->first] = it->second;
    m_coords.row(it->first).fill(datum::nan);
  }
}


TypeRegistry Coords::_find_oob(const uvec & grid_size) const{
  // Make the bounding box, and OOB rule
  mat bbox = zeros(n_dim,2);
  bbox.col(1) = grid_size - 1;
  OutOfBoundsRule oob_rule = OutOfBoundsRule(bbox, OOB_TYPE);

  // Apply the rule to get OOB rows
  return rule.type_element(m_coords);
}


void Coords::restrict_coords(const uvec & grid_size){
  /*
   * Check which coordinate rows violate the grid, and 
   * NaN them out.
   */

  // Apply the rule to get OOB rows
  TypeRegistry reg = _find_oob(grid_size);
  m_type_map.insert(reg.begin(), reg.end());  
  _mark(m_type_map);
}


uint Coords::number_of_spatial_coords() const{
  return number_of_all_coords() - number_of_special_coords();
}


uint Coords::number_of_all_coords() const{
  return m_coords.n_rows;
}


uint Coords::number_of_all_coords() const{
  return m_type_map.size();
}

uint Coords::max_spatial_index(const uvec & grid_size) const{
  return prod(grid_size) - 1;
}



/***************************************************************************
 * UNIFORM GRID CLASS *
 **********************/


UniformGrid::UniformGrid(vec & low,
			  vec & high,
			 uvec & num_cells) :
  m_low(low),
  m_high(high),
  m_num_cells(num_cells),
  m_num_nodes(num_cells + 1),
  m_width((high - low) / num_cells),
  n_dim(low.n_elem){
  uint D = low.n_elem;
  assert(D == high.n_elem);
  assert(D == num_cells);
}


TypedPoints UniformGrid::get_spatial_nodes(){
  /*
   * Later dimensions cycle faster, e.g.:
   * [0 0]
   * [0 0.1]
   * [0 0.2]
   * [1.5 0]
   * [1.5 0.1]
   *  ...
   */
  vector<vec> marginal_grids;
  for(uint d = 0; d < n_dim; d++){
    vec mg = linspace<vec>(m_low(d),
			   m_high(d),
			   num_num_nodes(d));
    if(mg.size() > 1){
      // Make sure the grid width is expected.
      assert(abs(mg(1) - mg(0) - m_width(d)) < PRETTY_SMALL);
    }
    marginal_grids.push_back(mg);
  }
  Points points = make_points(marginal_grids);
  assert(number_of_spatial_nodes() == points.n_rows);
  return TypedPoints(points);
}


TypedPoints UniformGrid::get_cell_centers(){\
  /*
   * Like above, but one less point per dimension, and shifted to the middle
   * of the range
   */
  vector<vec> marginal_grids;
  for(uint d = 0; d < n_dim; d++){
    vec mg = linspace<vec>(m_low(d) + 0.5 * m_width(d),
			   m_high(d) - 0.5 * m_width(d),
			   num_num_cells(d));
    marginal_grids.push_back(mg);
  }
  Points points = make_points(marginal_grids);
  assert(number_of_cells() == points.n_rows);
  return TypedPoints(points);
}

umat UniformGrid::get_cell_node_indices(){
  /*
   * Go through the cells in C order, and print the 
   * node ids associated with the cells.
   */
  vector<vec> marginal_coords;
  for(uint d = 0; d < n_dim; d++){
    uvec mc = regspace<uvec>(0,m_num_cells(d));
    marginal_coords.push_back(mc);
  }
  Points coord_points = make_points(marginal_coords);
  assert(number_of_cells() == coords.n_rows);
  Coords coords = Coords(m_num_points, coord_points);
  
  
}


uvec UniformGrid::cell_coords_to_low_node_indices(const Coords & coords){
  return Coords::_coords_to_indices(m_num_nodes, coords);
}


uvec UniformGrid::cell_coords_to_vertices(const Coords & coords){
  uint N = coords.number_of_all_coords();
  uint D = coords.n_dim;
  uint V = pow(2.0,D);
  assert(n_dim == D);
  
  uvec shift = c_order_cell_shift(m_num_nodes);
  assert(shift.n_elem == V);
  
  uvec low_idx = cell_coords_to_low_node_indices(coords);
  umat vertices = umat(N,V);

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


/***********************************************************************
 * OLD CODE REGION
 */

OutOfBounds UniformGrid::points_to_out_of_bounds(const Points & points){
  // Two passes: identify the OOB points, then classify
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(D == n_dim);

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
  assert(D == n_dim);

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


Points UniformGrid::cell_coords_to_low_node(const Coords & coords){
  Points low_points = Points(coords.num_total,coords.dim);
  low_points.rows(coords.oob.indices).fill(datum::nan);
  mat scaled = row_mult(conv_to<mat>::from(coords.coords),
			conv_to<rowvec>::from(m_width));
  low_points.rows(coords.indices) = row_add(scaled,
					    conv_to<rowvec>::from(m_low));
  return low_points;
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
				      
