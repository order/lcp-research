#include <iostream>
#include <assert.h>

#include "discrete.h"
#include "misc.h"

using namespace arma;

void print_grid(const RegGrid & grid){
  std::cout << grid.low << std::endl;
  std::cout << grid.high << std::endl;
  std::cout << grid.num_cells << std::endl;
}

uint num_actions(const mat & actions){
  return actions.n_rows;
}
uint action_dim(const mat &actions){
  return actions.n_cols;
}

uint num_states(const mat & states){
  return states.n_rows;
}
uint state_dim(const mat & states){
  return states.n_cols;
}

uint grid_dim(const RegGrid & grid){
  return grid.num_cells.n_elem;
}

uvec num_grid_points_per_dim(const RegGrid & grid){
  // Number of cells + 1: | * | * |
  return grid.num_cells + 1;
}
uint num_grid_points(const RegGrid & grid){
  return last_oob_index(grid) + 1;
}

uint first_oob_index(const RegGrid& grid){
  // Out of bound index is the last valid index
  return first_oob_index(num_grid_points_per_dim(grid));
}
uint first_oob_index(const uvec& points_per_dim){
  return prod(points_per_dim);
}

uint last_oob_index(const RegGrid& grid){
  // Out of bound index is the last valid index
  return last_oob_index(num_grid_points_per_dim(grid));
}
uint last_oob_index(const uvec& points_per_dim){
  uint D = points_per_dim.n_elem;
  return first_oob_index(points_per_dim) + 2*D - 1;
}

vec width(const RegGrid & grid){
  // The width of the hyperrectangle
  return (grid.high - grid.low) / grid.num_cells;
}

bool verify(const RegGrid & grid){
  uint D = grid.low.n_elem;
  assert(check_dim(grid,D));
  for(uint d = 0; d < D; d++){
    assert(grid.low(d) < grid.high(d));
  }
  return true;
}

bool check_dim(const RegGrid & grid, uint D){
  // Make sure all dimensions of a RegGrid are consistant
  return grid.low.n_elem == D
    && grid.high.n_elem == D
    && grid.num_cells.n_elem == D;
}


void build_oob(OOBInfo & oob, const mat & points, const RegGrid & grid){
  uint lo_oob = first_oob_index(grid);

  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(check_dim(grid,D));
  
  oob.mask = zeros<bvec>(N);
  int idx;
  for(uint i = 0; i < N; i++){
    //std::cout << "Point: " << points.row(i);
    idx = find_oob_index(points.row(i).t(),grid);
    //std::cout << "\tIdx: " << idx << std::endl;;
    if(idx > 0){
      oob.mask(i) = 1;
      oob.partial_map[i] = idx;
    }
  }
  //std::cout << "Mask:" << oob.mask << std::endl;
  oob.oob_idx = find(oob.mask == 1);
  oob.inb_idx = find(oob.mask == 0);
  
  oob.n_oob = oob.oob_idx.n_elem;
  oob.n_inb = oob.inb_idx.n_elem;

  //std::cout << "OOB (" << oob.n_oob << "): " << oob.oob_idx;
  //std::cout << "INB (" << oob.n_inb << "): " << oob.inb_idx;

}
int find_oob_index(const vec & state, const RegGrid & grid){
  uint D = state.n_elem;
  uint lo_oob = first_oob_index(grid);

  for(uint d = 0; d < D; d++){
    if(state(d) < grid.low(d)){
      return lo_oob + 2*d;
    }
    if(state(d) > grid.high(d) + GRID_FUDGE){
      return lo_oob + 2*d + 1;
    }
  }
  return -1;
}
vec get_oob_state(uint oob_idx, const RegGrid & grid){
  assert(oob_idx >= first_oob_index(grid));
  assert(oob_idx <= last_oob_index(grid));  

  vec state = 0.5 * (grid.high + grid.low); // Midpoint
  uint d = (oob_idx - first_oob_index(grid)) / 2;
  assert(d <= state.n_elem);

  if(0 == oob_idx % 2){
    state(d) = grid.low(d) - 0.1 * (grid.high(d) - grid.low(d));
  }
  else{
    state(d) = grid.high(d) + 0.1 * (grid.high(d) - grid.low(d));
  }
  assert(oob_idx == find_oob_index(state,grid));
  return state;
}

vector<vec> vectorize(const RegGrid & grid){
  // Convert grid description (lo,hi,num) into
  // actual points [low,low+delta,...,hi]
  vector<vec> grids;
  uint D = grid.low.n_elem;
  for(uint d = 0; d < D; d++){
    grids.push_back(linspace<vec>(grid.low(d),
				  grid.high(d),
				  grid.num_cells(d)+1));
  }
  return grids;
}

mat make_points(const RegGrid & grid){
  return make_points(vectorize(grid));
}

mat make_points(const vector<vec> & grids)
{
  // Makes a mesh out of the D vectors
  // 'C' style ordering... last column changes most rapidly
  
  // Figure out the dimensions of things
  uint D = grids.size();
  uint N = 1;
  for(vector<vec>::const_iterator it = grids.begin();
      it != grids.end(); ++it){
    N *= it->n_elem;
  }
  mat P = mat(N,D); // Create the matrix
  
  uint rep_elem = N; // Element repetition
  uint rep_cycle = 1; // Pattern rep
  for(uint d = 0; d < D; d++){
    uint n = grids[d].n_elem;
    rep_elem /= n;
    assert(N == rep_cycle * rep_elem * n);
    
    uint I = 0;
    for(uint c = 0; c < rep_cycle; c++){ // Cycle repeat
      for(uint i = 0; i < n; i++){ // Element in pattern
	for(uint e = 0; e < rep_elem; e++){ // Element repeat
	  assert(I < N);
	  P(I,d) = grids[d](i);
	  I++;
	}
      }
    }
    rep_cycle *= n;
  }
  return P;
}

uvec c_order_coef(const RegGrid & grid){
  return c_order_coef(num_grid_points_per_dim(grid));
}

uvec c_order_coef(const uvec & points_per_dim){
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

   Not the other ordering is "Fortran order":

   2 - 5
   |   |
   1 - 4
   |   |
   0 - 3

   define by coefficients [1,3]. We're not using these.
  */
  
  uint D = points_per_dim.n_elem;
  uvec coef = uvec(D);

  uint agg = 1;
  for(uint i = D; i > 0; --i){
    coef(i-1) = agg;
    agg *= points_per_dim(i-1);
  }
  return coef;
}

uvec cell_shift_coef(const RegGrid & grid){
  return cell_shift_coef(num_grid_points_per_dim(grid));
}

uvec cell_shift_coef(const uvec & points_per_dim){
  /* The index shift for the other vertices of the hyper rectangle
     So if the cell shift coef are [9 3 1] (from a (? x 3 x 3) grid)

     4----13
    /|   /|
   3-+--12|
   | |  | |
   | 1--+-10
   |/   |/
   0----9

  */
  
  uint D = points_per_dim.n_elem;
  uvec coef = c_order_coef(points_per_dim);
  
  uint V = pow(2,D);  
  uvec shift = zeros<uvec>(V);
  bvec b;
  for(uint d = 0; d < D; ++d){
    b = binmask(d,D);
    shift(find(b == 1)) += coef[d];
  }
  return shift;
}

uvec coords_to_indices(const umat & coords,
		       const RegGrid & grid,
		       const OOBInfo & oob){
  return coords_to_indices(coords,
			   num_grid_points_per_dim(grid),
			   oob);
}
uvec coords_to_indices(const umat & coords,
		       const uvec & points_per_dim,
		       const OOBInfo & oob){
  // Converts a matrix of coordinates to indices
  
  uvec coef = c_order_coef(points_per_dim);
  uvec idx = uvec(coords.n_rows);
  idx.rows(oob.inb_idx) = coords.rows(oob.inb_idx) * coef;
  for(oob_map::const_iterator it = oob.partial_map.begin();
      it != oob.partial_map.end(); it++){
    idx(it->first) = it->second;
  }  
  return idx;
}

umat indices_to_coords(const uvec & idx, const uvec & points_per_dim){
  uvec coef = c_order_coef(points_per_dim);
  
  uint D = points_per_dim.n_elem;
  uint N = idx.n_elem;
  
  umat crd = umat(N,D);
  uvec curr_idx = idx;
  for(uint d = 0; d < D; ++d){
    crd.col(d) =  floor(curr_idx / coef(d));
    curr_idx = vec_mod(curr_idx, coef(d));
  }

  // Ignore OOB indicies; will map directly
  uvec oob_idx = find(idx >= first_oob_index(points_per_dim));
  crd.rows(oob_idx).fill(OOB_COORD);
  
  return crd;
}

umat least_coord(const mat & points,
		 const RegGrid & grid,
		 const OOBInfo & oob){
  /*
    Returns the actual coordinate of the least cutpoint in
    the hypercube that the point is in.
   */
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(N > 0);
  assert(check_dim(grid,D));

  // Do mat as doubles then convert to umat
  mat scaled = mat(N,D);
  
  if(oob.n_inb > 0){
    // Shift by l
    mat diff = row_diff(points.rows(oob.inb_idx),grid.low.t());
    
    // Scale by n / (h - l) (inverse width)
    rowvec scale = conv_to<rowvec>::from(grid.num_cells)
      / (grid.high.t() - grid.low.t());    
    scaled.rows(oob.inb_idx) = row_mult(diff,scale);
  }
  if(oob.n_oob > 0){
    // Ignore oob points
    scaled.rows(oob.oob_idx).fill(OOB_COORD);
  }

  // Take floor; rounds down to least coord
  umat cuts = conv_to<umat>::from(floor(scaled));
  return cuts;
}

void point_to_idx_dist(const mat & points,
			 const RegGrid & grid,
			 sp_mat & out_spmat){


  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(check_dim(grid,D));

  // Get the width of the rectangles
  rowvec delta = conv_to<rowvec>::from(width(grid));

  // Get the oob points; deal with seperately
  // Assumption: there aren't enough of these to make handling them
  // cleverly worth it.
  OOBInfo oob;  
  build_oob(oob,points,grid);

  // Get the least vertex for the cell enclosing each point
  umat low_coords = least_coord(points,grid,oob);
  
  /*
    Want to get  distance from least cell cut points (denoted by the @ below):
      o-----o
     / |   /|
    o--+--o |
    |  o--|-o
    | / x |/ 
    @-----o

    We want the distance normalized by cell size (delta vector):
    P / d - (C + (l /d)) = (P - (l + C*d)) / d
   */
  mat norm_points = points.each_row() / delta; // (P / d)
  rowvec norm_low = grid.low.t() / delta; // (l / d)
  mat norm_least_pos = (conv_to<mat>::from(low_coords)).each_row() + norm_low;
  //C + (l / d)
  mat norm_diff = norm_points - norm_least_pos; // Whole thing

  // Calculate multi-linear weights over the cube vertices
  uint V = pow(2,D); // Number of points per hyperrect
  mat W = ones<mat>(N,V);
  uint halfV = pow(2,D-1); // Half the number of points (one side)

  /* Iterate through the dimensions
     Update half with the distance from the low side of the cube
     Update the rest with the distance from the far side */
  for(uint d = 0; d < D; ++d){
    bvec mask = binmask(d,D);
    W.cols(find(mask == 1)) %=
      repmat(norm_diff.col(d),1,halfV);
    W.cols(find(mask == 0)) %=
      repmat(1.0 - norm_diff.col(d),1,halfV);
  }

  // Calculate the index of the low grid point
  uvec low_indices = coords_to_indices(low_coords,grid,oob);
  
  // Calculate the shift for the regular grid cells
  uvec shift = cell_shift_coef(grid);

  // Build the sparse matrix
  uint sp_nnz = oob.n_inb * V + oob.n_oob; // Number of nnz in sp_matrix
  vec data = vec(sp_nnz);
  umat loc = umat(2,sp_nnz); // Location pairs (i,j)

  // Fill in loc matrix
  uint I = 0;
  
  // Fill in data and locations
  uvec sort_idx = sort_index(shift); // Done to avoid sorting in SP creation
  uint J;
  for(uint i = 0; i < N; i++){ // Iterate over states
    for(uint j = 0; j < V; j++){ // Iterate over vertices
      if (oob.mask(i) == 1 && j > 0){
	// Already filled in oob entry
	continue;
      }
      assert(I < sp_nnz);
      if(oob.mask(i) == 1 && j == 0){
	// OOB and first occurance
	data(I) = 1.0;
	loc(0,I) = oob.partial_map[i];
	loc(1,I++) = i;
      }
      if(oob.mask(i) == 0){
	// In bounds
	J = sort_idx(j);
	data(I) = W(i,J);
	loc(0,I) = low_indices(i) + shift(J);
	loc(1,I++) = i;
      }      
    }
  }
  uint G = num_grid_points(grid);
  //std::cout << "Low indices:" << low_indices.t();
  //std::cout << "Shift:" << shift.t();
  
  //std::cout << "Dimensions (" <<G << ',' << N << ")\n";
  //std::cout << "Loc:\n" << loc.t();
  //std::cout << "Data:\n" << data;
  
  out_spmat = sp_mat(loc,data,G,N,false);
}

vec interp_fn(const vec & vals, const mat & points,const RegGrid & grid){
  return interp_fns(vals,points,grid); // Should cast appropriately
}

mat interp_fns(const mat & vals,
	       const mat & points,
	       const RegGrid & grid){
  uint G = num_grid_points(grid);
  uint N = points.n_rows;
  uint D = points.n_cols;
  uint M = vals.n_cols;
  
  assert(check_dim(grid,D));
  assert(G == vals.n_rows);

  sp_mat point_dist;
  point_to_idx_dist(points,grid,point_dist);
  assert(G == point_dist.n_rows);
  assert(N == point_dist.n_cols);  
  
  mat I = (vals.t() * point_dist).t();
  assert(N == I.n_rows);
  assert(M == I.n_cols);

  return I;
}

uvec max_interp_fns(const mat & vals,
		    const mat & points,
		    const RegGrid & grid){
  uint N = points.n_rows;
  mat I = interp_fns(vals,points,grid);
  uvec idx = col_argmax(I);
  assert(N == idx.n_elem);
  return idx;
}
uvec min_interp_fns(const mat & vals,
		    const mat & points,
		    const RegGrid & grid){
  return max_interp_fns(-vals,points,grid);
}

