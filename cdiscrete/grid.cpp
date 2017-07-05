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
     Coeffs to converts from grid coords to indices
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
   
     NB: For 2 and 3 dim, we check with sub2ind. Since armadillo is
     column-major, we need to do some flipping.
  */
  
  uint D = grid_size.n_elem;
  uvec stride = flipud(shift(cumprod(flipud(grid_size)),1));
  stride(stride.n_elem - 1) = 1;
  
#ifndef NDEBUG
  // Check for 2 and 3 dimensions
  if(2 == D){
    SizeMat sz = uvec2sizemat(flipud(grid_size));
    umat I = eye<umat>(2,2);
    uvec stride_check = flipud(sub2ind(sz, I));
    assert(all(stride_check == stride));
  }
  else if(3 == D){
    SizeCube sz = uvec2sizecube(flipud(grid_size));
    umat I = eye<umat>(3,3);
    uvec stride_check = flipud(sub2ind(sz, I));
    assert(all(stride_check == stride));
  }
#endif
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
     This pattern, <0,3,1,4> is the "C-order cell shift".

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




Coords indices_to_coords(const uvec & grid_size,
		       const uvec & indices){
  /*
   * Convert indices into coordinates by repeated modulus.
   * NB: makes sense because c-order stride is in decreasing order
   */

  uint n_dim = grid_size.n_elem;
  uvec stride = c_order_stride(grid_size);
  imat raw_coords = imat(indices.n_elem, n_dim);

  umat mod_res = umat(indices.n_elem, 2);
  mod_res.col(1) = indices;
  for(uint d = 0; d < n_dim; d++){
    mod_res = divmod(mod_res.col(1), stride(d));
    raw_coords.col(d) = conv_to<ivec>::from(mod_res.col(0));
  }

  // Build the type registry
  TypeRegistry reg;
  uint first_special_index = prod(grid_size);
  uvec non_spatial = find(indices >= first_special_index);
  for(auto const & it : non_spatial){
    reg[it] = indices[it] - first_special_index + 1;
  }

  Coords coords = Coords(raw_coords, reg);
  assert(coords.check(grid_size));
  return coords;
}


uvec coords_to_indices(const uvec & grid_size,
		       const Coords & coords){
  // Calculate indices for normal coords
  assert(coords.check(grid_size));
  
  // TODO: convert coords to uvec somehow
  uvec stride = c_order_stride(grid_size);
  umat indices = conv_to<umat>::from(coords.m_coords) * stride; 

  // OOB special indices
 
  uint first_special_idx = prod(grid_size);
  for(auto const & it : coords.m_reg){
    assert(Coords::SPATIAL_TYPE != it.second);
    indices(it.first) = it.second + first_special_idx - 1;
  }
  assert(is_finite(indices));  
  return indices;
}


/************************************************************************
 * COORDINATE CLASS *
 ********************/

Coords::Coords(const imat & coords){
  m_coords = coords;
  n_rows = coords.n_rows;
  n_dim = coords.n_cols;

  uvec special = get_special();
  _mark(special, Coords::DEFAULT_OOB_TYPE);
  assert(check());
}

Coords::Coords(const imat & coords, const TypeRegistry & reg){
  m_coords = coords;
  n_rows = coords.n_rows;
  n_dim = coords.n_cols;

  _mark(reg);
  assert(check());
}

   
bool Coords::check() const {
  uvec sp_idx = get_special();
  if(sp_idx.n_elem != m_reg.size()) return false;
  for(auto const& it : sp_idx){
    if(!is_special(it)) return false;
  }
  return true;
}


bool Coords::check(const uvec & grid_size) const{
  return check() && _coord_check(grid_size);
}


bool Coords::_coord_check(const uvec & grid_size) const{
  /*
   * Check to make sure that the coordinates make sense with the supplied
   * grid sizes.
   */
  for(uint d = 0; d < n_dim; d++){
    if(any(m_coords.col(d) >= grid_size(d))) return false;
  }
  return true;
}



void Coords::_mark(const uvec & indices, uint coord_type){
  /*
   * Go through and add the indices to the type map with the supplied
   * coord_type
   */
  assert(coord_type > Coords::SPATIAL_TYPE);
  for(auto const & it : indices){
    m_reg[it] = coord_type;
    m_coords.row(it).fill(Coords::SPECIAL_FILL);
  }

  assert(check());
}


void Coords::_mark(const TypeRegistry & reg){
  /*
   * Go through and add the indices to the type map with the supplied
   * coord_type
   */
  for(auto const & it : reg){
    assert(it.second != Coords::SPATIAL_TYPE);
    m_reg[it.first] = it.second;
    m_coords.row(it.first).fill(Coords::SPECIAL_FILL);
  }
  assert(check());
}


TypeRegistry Coords::_find_oob(const uvec & grid_size, uint type) const{
  assert(type > Coords::SPATIAL_TYPE);
  // Make the bounding box, and OOB rule
  mat bbox = zeros(n_dim,2);
  bbox.col(1) = conv_to<vec>::from(grid_size) - 1;
  OutOfBoundsRule oob_rule = OutOfBoundsRule(bbox, type);

  // Apply the rule to get OOB rows
  return oob_rule.type_elements(conv_to<mat>::from(m_coords));
}


uint Coords::num_spatial() const{
  return num_coords() - num_special();
}


uint Coords::num_coords() const{
  return m_coords.n_rows;
}


uint Coords::num_special() const{
  return m_reg.size();
}

uint Coords::max_spatial_index(const uvec & grid_size) const{
  return prod(grid_size) - 1;
}
bool Coords::is_special(uint idx) const{
  return m_reg.end() != m_reg.find(idx);
}

uvec Coords::get_indices(const uvec & grid_size) const{
  return coords_to_indices(grid_size, *this);
}

uvec Coords::get_spatial() const{
  return find(m_coords.col(0) >= 0);
}
uvec Coords::get_special() const{
  return find(m_coords.col(0) < 0);
}

bool Coords::equals(const Coords & other) const{
  // Check dimensions
  if(other.n_rows != this->n_rows) return false;
  if(other.n_dim != this->n_dim) return false;
  if(other.m_reg.size() != this->m_reg.size()) return false;

  // Check the registry
  for(auto const& it : other.m_reg){
    uint idx = it.first;
    uint val = it.second;
    if(this->m_reg.end() == this->m_reg.find(idx)) return false;
    if(val != this->m_reg.at(idx)) return false;
  }

  // Check the coords
  return all(all(other.m_coords == this->m_coords));
}

ostream& operator<<(ostream& os, const Coords& c){
  for(uint i = 0; i < c.num_coords(); i++){
    os << "Node [" << i << "]:";
    if(c.is_special(i)){
      os << "\tSpecial (" << c.m_reg.find(i)->second << ")" << endl;
    }
    else{
      os << c.m_coords.row(i);
    }
  }
  return os;
}



/***************************************************************************
 * UNIFORM GRID CLASS *
 **********************/


UniformGrid::UniformGrid(vec & low,
			 vec & high,
			 uvec & num_cells,
			 uint special_nodes) :
  m_low(low),
  m_high(high),
  m_num_cells(num_cells),
  m_num_nodes(num_cells + 1),
  m_width((high - low) / num_cells),
  n_dim(low.n_elem),
  n_special_nodes(special_nodes){
  uint D = low.n_elem;
  assert(D == high.n_elem);
  assert(D == num_cells.n_elem);
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
  assert(number_of_cells() == coord_points.n_rows);
  
  Coords coords = Coords(conv_to<imat>::from(coord_points));
  assert(coords.check());
  assert(0 == coords.num_special());
  return coords.get_indices(m_num_cells);
}

uint UniformGrid::number_of_all_nodes() const{
  return prod(m_num_nodes);
}

uint UniformGrid::number_of_spatial_nodes() const{
  return n_special_nodes;
}

uint UniformGrid::number_of_cells() const{
  return prod(m_num_cells);
}


uint UniformGrid::max_spatial_node_index() const{
  return prod(m_num_nodes) - 1;
}

uint UniformGrid::max_node_index() const{
  return max_spatial_node_index() + n_special_nodes;
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
  uint N = coords.num_coords();
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
  uint oob_idx = max_spatial_node_index();
  for(auto const & it : coords.m_reg){
    assert(it.second > 0);
    vertices.row(it.first).fill(oob_idx + it.second);
  }
  assert(!vertices.has_nan()); // All should be dealt with.
  
  return vertices;
}

Coords UniformGrid::points_to_cell_coords(const TypedPoints & points) const{
  // Takes in points, spits out cell coords
  assert(points.check_in_bbox(m_low,m_high));

  // C = floor((P - low) / width)
  mat diff = row_diff(points.m_points, conv_to<rowvec>::from(m_low));
  mat scaled = row_divide(diff, conv_to<rowvec>::from(m_width));
  imat raw_coords = conv_to<imat>::from(floor(scaled));

  // Use all existing typing information.
  return Coords(raw_coords, points.m_reg); // Use points information.
}

TypedPoints UniformGrid::cell_coords_to_low_points(const Coords & coords) const{
  // Reverse of above
  mat scaled = row_mult(conv_to<mat>::from(coords.m_coords), m_width);
  mat raw_points = row_add(scaled, m_low);
  return TypedPoints(raw_points, coords.m_reg);
}



mat UniformGrid::points_to_cell_nodes_dist(const TypedPoints & points,
					   const Coords & coords) const{
  /*
   * Takes in points, and returns a matrix with the distances to the 
   */
  assert(n_dim == points.m_points.n_cols);
  assert(points.check_in_bbox(m_low,m_high));
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


ElementDist UniformGrid::points_to_element_dist(const TypedPoints & points)
  const{
  /*
   * Does multi-linear interpolation. Points internal to the cell are mapped
   * to a probability distribution over vertices of that cell, with weights
   * depending on distance from the vertex.
   * So the weight of the (x) point will be largest for the bottom-left 
   * corner, and least for the top-right corner.
   o----o
   |    |
   | x  |
   o----o
   */
  
  // Assume points are properly bounded.
  assert(points.check_in_bbox(m_low, m_high));

  Coords coords = points_to_cell_coords(points);
  mat dist = points_to_cell_nodes_dist(points, coords);
  mat rel_dist = row_divide(dist,m_width);

  uint N = points.m_points.n_rows;
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
  assert(is_finite(weights));
  assert(all(all(weights >= 0)));
  assert(all(all(weights <= 1)));
  
  // Check the weights of the low and high nodes.
  // position 0 -> 00...0 is the low node, so weight should be prod(rel_dist)
  // position V-1 -> 11...1, so weight should be prod(1-rel_dist)
  vec low_node_weights = prod(1 - rel_dist,1);
  vec hi_node_weights = prod(rel_dist,1);
  assert(approx_equal(weights.col(0),low_node_weights,"absdiff",1e-12));
  assert(approx_equal(weights.col(V-1),hi_node_weights,"absdiff",1e-12));

  // Sparsify
  for(uint d = 0; d < D; d++){
    weights(find(weights.col(d) <= ALMOST_ZERO),
	    uvec({d})).fill(0);
  }

  // Convert into a sparse matrix.
  umat vert_indices = cell_coords_to_vertex_indices(coords);
  uint n_nodes = max_node_index() + 1;
  ElementDist distrib = build_sparse_dist(n_nodes, vert_indices, weights);

  return distrib; 
}


template <typename T> T UniformGrid::base_interpolate(const TypedPoints & points,
						      const T& data) const{  
  uint N = points.n_rows;
  uint d = points.n_cols;
  uint NN = max_node_index() + 1;
  assert(data.n_rows == NN);
  
  ElementDist D = points_to_element_dist(points);
  assert(size(D) == size(NN,N));

  T ret = D.t() * data;
  assert(ret.n_rows == N);
  assert(ret.n_cols == data.n_cols);
  return ret;
}

vec UniformGrid::interpolate(const TypedPoints & points,
			     const vec & data) const{
  return base_interpolate<vec>(points,data);
}

mat UniformGrid::interpolate(const TypedPoints & points,
			     const mat & data) const{
  return base_interpolate<mat>(points,data);
}

mat UniformGrid::find_bounding_box() const{
  mat bounds = mat(n_dim,2);
  bounds.col(0) = m_low;
  bounds.col(1) = m_high;
  assert(check_bbox(bounds));
  return bounds;
}

mat UniformGrid::cell_gradient(const vec & value) const{
  assert(false); // TODO: fill this in.
  return mat();
}



ElementDist build_sparse_dist(uint n_nodes, umat vert_indices, mat weights){
  // Take in the number of nodes, the vertex indices, and the weights
  // and build the probabilistic transition matrix.
  vector<uword> v_row;
  vector<uword> v_col;
  vector<double> v_data;

  uint N = vert_indices.n_rows;
  uint V = vert_indices.n_cols;

  // Basic sanity checking.
  assert(is_finite(weights));
  assert(all(all(weights >= 0)));
  assert(all(all(weights <= 1)));
  assert(all(all(vert_indices < n_nodes)));

  // Build out the row, column, and data vectors.
  for(uint i = 0; i < N; i++){
    uvec uni = unique(vert_indices.row(i));
    assert(uni.n_elem == 1 || uni.n_elem == V);
    if(1 == uni.n_elem){
      // Transition to non-spatial node.
      // Currently assume that there is a unique special node with all the
      // weight.
      assert(norm(weights(i,0) - 1) < PRETTY_SMALL); // All weight
      v_row.push_back(i);
      v_col.push_back(uni(0));
      v_data.push_back(1.0);
    }
    else{
      assert(V == uni.n_elem); // Proper spatial dist
      assert(norm(weights.row(i) - 1) < PRETTY_SMALL); // Transition
      for(uint j = 0; j < V; j++){
	v_row.push_back(i);
	v_col.push_back(vert_indices(i,j));
	v_data.push_back(weights(i,j));
      }
    }
  }

  uint nnz = v_row.size();
  umat loc = umat(2,nnz);
  loc.row(0) = urowvec(v_row);
  loc.row(1) = urowvec(v_col);
  vec data = vec(v_data);
  assert(abs(sum(data) - N) < PRETTY_SMALL);
  return sp_mat(loc,data,N,n_nodes);
}
				      
