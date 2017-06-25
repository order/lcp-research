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

     TODO: better comment explaining this.
  */
  
  uint D = points_per_dim.n_elem;
  uvec strides = c_order_stride(points_per_dim);
  uint V = pow(2,D);
  
  uvec shifts = zeros<uvec>(V);
  uvec idx;
  for(uint d = 0; d < D; d++){
    idx = find(binmask(d,D)); // Find indices where bit 1 is lit
    assert(idx.n_elem <= V);
    shifts(idx) += strides(d); // Add stride d to those locations
  }
  return shifts;
}


/************************************************************************
 * COORDINATE CLASS *
 ********************/


Coords::Coords(const umat & coords){
  m_coords = coords;
  n_rows = coords.n_rows;
  n_dim = coords.n_cols;

  uvec special = find_nonfinite(coords.col(0));
  _mark(special, Coords::OOB_TYPE);
}

Coords::Coords(const umat & coords, const TypeRegistry & reg){
  m_coords = coords;
  n_rows = coords.n_rows;
  n_dim = coords.n_cols;

  _mark(reg);
}



bool Coords::check(const uvec & grid_size) const{
  assert(_coord_check(grid_size, m_coords));
  assert(_type_reg_check(m_reg, m_coords));
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
  assert(coords.check(grid_size));
  
  // TODO: convert coords to uvec somehow
  uvec stride = c_order_stride(grid_size);
  umat indices = conv_to<umat>::from(coords.m_coords) * stride; 

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
    m_reg[it] = coord_type;
    m_coords.row(it).fill(datum::nan);
  }
  assert(_type_reg_check(m_reg, m_coords));
}


void Coords::_mark(const TypeRegistry & reg){
  /*
   * Go through and add the indices to the type map with the supplied
   * coord_type
   */
  for(auto const & it : reg){
    assert(it.second != SPATIAL_TYPE);
    m_reg[it.first] = it.second;
    m_coords.row(it.first).fill(datum::nan);
  }
  assert(_type_reg_check(m_reg, m_coords));
}


TypeRegistry Coords::_find_oob(const uvec & grid_size) const{
  // Make the bounding box, and OOB rule
  mat bbox = zeros(n_dim,2);
  bbox.col(1) = conv_to<vec>::from(grid_size) - 1;
  OutOfBoundsRule oob_rule = OutOfBoundsRule(bbox, OOB_TYPE);

  // Apply the rule to get OOB rows
  return oob_rule.type_elements(conv_to<mat>::from(m_coords));
}


uint Coords::number_of_spatial_coords() const{
  return number_of_all_coords() - number_of_special_coords();
}


uint Coords::number_of_all_coords() const{
  return m_coords.n_rows;
}


uint Coords::number_of_special_coords() const{
  return m_reg.size();
}

uint Coords::max_spatial_index(const uvec & grid_size) const{
  return prod(grid_size) - 1;
}

uvec Coords::get_indices(const uvec & grid_size) const{
  return _coords_to_indices(grid_size, *this);
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


TypedPoints UniformGrid::get_spatial_nodes() const{
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
			   m_num_nodes(d));
    if(mg.size() > 1){
      // Make sure the grid width is expected.
      assert(abs(mg(1) - mg(0) - m_width(d)) < PRETTY_SMALL);
    }
    marginal_grids.push_back(mg);
  }
  mat points = make_points<mat,vec>(marginal_grids);
  assert(number_of_spatial_nodes() == points.n_rows);
  return TypedPoints(points);
}


TypedPoints UniformGrid::get_cell_centers() const{
  /*
   * Like above, but one less point per dimension, and shifted to the middle
   * of the range
   */
  vector<vec> marginal_grids;
  for(uint d = 0; d < n_dim; d++){
    vec mg = linspace<vec>(m_low(d) + 0.5 * m_width(d),
			   m_high(d) - 0.5 * m_width(d),
			   m_num_cells(d));
    marginal_grids.push_back(mg);
  }
  mat points = make_points<mat,vec>(marginal_grids);
  assert(number_of_cells() == points.n_rows);
  return TypedPoints(points);
}

umat UniformGrid::get_cell_node_indices() const{
  /*
   * Go through the cells in C order, and print the 
   * node ids associated with the cells.
   */
  vector<uvec> marginal_coords;
  for(uint d = 0; d < n_dim; d++){
    uvec mc = regspace<uvec>(0,m_num_cells(d));
    marginal_coords.push_back(mc);
  }
  umat coord_points = make_points<umat,uvec>(marginal_coords);
  assert(number_of_cells() == coords.n_rows);
  
  Coords coords = Coords(coord_points);
  assert(0 == coords.number_of_special_coords());
  return coords.get_indices(m_num_cells);
}


uvec UniformGrid::cell_coords_to_low_node_indices(const Coords & coords) const
{
  return coords.get_indices(m_num_nodes);
}


umat UniformGrid::cell_coords_to_vertex_indices(const Coords & coords) const{
  /*
   * Build a matrix where each row corresponds to the indices for the 
   * vertices of the coordinates.
   * E.g. for the grid:
     6 - 7 - 8
     | * |   |
     3 - 4 - 5
     |   |   |
     0 - 1 - 2
     The coordinate noted with the star, (0,1),  will be mapped to the row 
     [3,6,4,7].

     Out of bound nodes, e.g. (2,3) here, are mapped to the row of all oob
     indices. In this case, [9,9,9,9]
   */

  // Calc dimensions and sizes
  uint N = coords.number_of_all_coords();
  uint D = coords.n_dim;
  uint V = pow(2.0,D);
  assert(n_dim == D);

  // Get the index of the least indexed node in each cell
  uvec low_idx = cell_coords_to_low_node_indices(coords);

  // Get the static "shift" pattern
  uvec shift = c_order_cell_shift(m_num_nodes);
  assert(shift.n_elem == V);

  // Build the matrix of cell vertices
  umat vertices = umat(N,V);
  for(uint v = 0; v < V; v++){
    vertices.col(v) = low_idx + shift(v);
  }

  // Encode the oob row as all oob index
  uint oob_idx = max_spatial_cell_index();
  for(auto const & it : m_reg){
    assert(it.second > 0);
    vertices.row(it.first).fill(oob_idx + it.second);
  }
  assert(!vertices.has_nan()); // All should be dealt with.
  
  return vertices;
}

uint UniformGrid::max_spatial_cell_index() const{
  return prod( m_num_cells);
}


Coords UniformGrid::points_to_cell_coords(const TypedPoints & points) const{
  // Takes in points, spits out cell coords
  assert(points.check_bounding_box(m_low,m_high));

  // C = floor((P - low) / width)
  mat diff = row_diff(points.m_points, conv_to<rowvec>::from(m_low));
  mat scaled = row_divide(diff, conv_to<rowvec>::from(m_width));
  umat raw_coords = conv_to<umat>::from(floor(scaled));

  // Use all existing typing information.
  return Coords(raw_coords, points.m_reg); // Use points information.
}

TypedPoints UniformGrid::cell_coords_to_low_points(const Coords & coords) const{
  // Reverse of above
  mat scaled = row_mult(conv_to<mat>::from(coords.m_coords), m_width);
  mat raw_points = row_add(scaled, m_low);
  return TypedPoints(raw_points, coords.m_reg);
}



mat UniformGrid::points_to_cell_nodes_rel_dist(const TypedPoints & points,
					       const Coords & coords) const{
  /*
   * Takes in points, and returns a matrix with the distances to the 
   */
  assert(n_dim == points.n_cols);
  assert(points.check_bounding_box(m_low,m_high));
  uint N = points.m_points.n_rows;

  // Calculate
  TypedPoints low_points = cell_coords_to_low_points(coords);
  mat low_diff = points.m_points - low_points.m_points;

  uint V = pow(2.0,n_dim);
  mat dist = mat(N, V);
  for(uint v = 0; v < V; v++){
    // Build the delta vector; the difference from the low point
    uvec idx = find(num2binvec(v, n_dim));
    assert(idx.n_elem <= V);    
    vec delta = zeros(N);
    delta(idx) = m_width(idx);
    
    mat diff = row_add(low_diff, delta); // Add delta to customize it
    dist.col(v) = lp_norm(diff, 2, 1); // 2-norm done row-wise (the 1)
  }

  return dist;
}

// **************************************************************************


ElementDist UniformGrid::points_to_element_dist(const TypedPoints & points){
  // Assume points are properly bounded.
  assert(points.check_bounding_box(m_low, m_high));

  Coords coords = points_to_cell_coords(points);
  mat dist = points_to_cell_nodes_dist(points, coords);
  mat rel_dist = row_divide(dist,m_width);

  uint N = points.n_rows;
  uint D = n_dim;
  uint V = pow(2.0,D);
  uint halfV = pow(2.0,D-1);

  // Calculate the interpolation weights for each point
  mat weights = ones<mat>(N,V);
  bvec mask;
  mat rep_dist;
  for(uint d = 0; d < D; d++){
    rep_dist = repmat(rel_dist.col(d),1, halfV);
    mask = binmask(d,D);   
    weights.cols(find(mask == 1)) %= rep_dist;		       
    weights.cols(find(mask == 0)) %= 1 - rep_dist;
  }
  assert(all(weights >= 0));
  assert(all(weights <= 1));
  assert(!weights.has_nan());
  
  // Check the weights of the low and high nodes.
  // position 0 -> 00...0 is the low node, so weight should be prod(rel_dist)
  // position V-1 -> 11...1, so weight should be prod(1-rel_dist)
  vec low_node_weights = prod(1 - rel_dist.rows(inb_idx),1);
  vec hi_node_weights = prod(rel_dist.rows(inb_idx),1);
  assert(approx_equal(weights.col(0),low_node_weights,"absdiff",1e-12));
  assert(approx_equal(weights.col(V-1),hi_node_weights,"absdiff",1e-12));

  // Sparsify
  for(uint d = 0; d < D; d++){
    weights(find(weights.col(d) <= ALMOST_ZERO),
	    uvec({d})).fill(0);
  }

  // Convert into a sparse matrix.
  umat vert_indices = cell_coords_to_vertex_indices(coords);
  ElementDist distrib = pack_weights(weights, vert_indices);

  return distrib; 
}

ElementDist pack_vertices_and_weights(mat weights, umat vert_indices){
  // TODO:
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
				      
