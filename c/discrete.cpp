#include <iostream>
#include <assert.h>
#include "discrete.h"

#include <chrono>


using namespace std;
using namespace std::chrono;
using namespace arma;

uvec num_grid_points_per_dim(const RegGrid & grid){
  // Number of cells + 1: | * | * |
  return grid.num_cells + 1;
}
uint num_grid_points(const RegGrid & grid){
  // Product, plus an addition
  return prod(grid.num_cells + 1) + 1;
}

uint oob_index(const RegGrid& grid){
  // Out of bound index is the last valid index
  return oob_index(grid.num_cells + 1);
}
uint oob_index(const uvec& lens){
  // Lens = number of points per dimension
  return prod(lens);
}

vec width(const RegGrid & grid){
  // The width of the hyperrectangle
  return (grid.high - grid.low) / grid.num_cells;
}

bool check_dim(const RegGrid & grid, uint D){
  // Make sure all dimensions of a RegGrid are consistant
  return grid.low.n_elem == D
    && grid.high.n_elem == D
    && grid.num_cells.n_elem == D;
}

void generate_mask(Mask & M, bvec & source){
  M.mask = source;
  M.pos = find(source == 1);
  M.neg = find(source == 0);
  M.n_pos = M.pos.n_elem;
  M.n_neg = M.neg.n_elem;
}

uvec vec_mod(const uvec & a, uint n){
  return a - (a / n) * n;
}

bvec num2binvec(uint n, uint D){
  // Convert n to a D-bit binary vector
  // Little endian
  assert(n < pow(2,D));
  bvec b = bvec(D);
  for(uint d = 0; d < D; ++d){
    b(d) = ((1 << d) & n) >> d;
  }
  return b;
}

bvec binmask(uint d, uint D){
  // For all of numbers in 0,..., 2**D-1, do they have bit d lit up?
  // e.g. binmask(0,D) will be: [0,1,0,1,...]
  // e.g. binmask(1,D) will be: [0,0,1,1,0,0,1,1,...]

  uint N = pow(2,D);
  bvec mask = bvec(N);
  for(uint b = 0; b < N; ++b){
    mask[b] = ((1 << d) & b) >> d; // Shift 1 over, mask, then shift back.
  }
  return mask;
}

vector<vec> vectorize(const RegGrid & grid){
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
  for(int d = 0; d < D; d++){
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

uvec c_order_coef(const uvec & lens){
  // Coeffs to convert coords to indicies (inline?)
  uint D = lens.n_elem;
  uvec coef = uvec(D);

  uint agg = 1;
  for(uint i = D; i > 0; --i){
    coef(i-1) = agg;
    agg *= lens(i-1);
  }
  return coef;
}

uvec cell_shift_coef(const uvec & lens){
  // The index shift for the vertices of the hyper rectangle
  // Basically, Hamming distance, but using c_order_coefs rather
  // than ones(D) as the `weight' of each dimension.
  
  uint D = lens.n_elem;
  uvec coef = c_order_coef(lens);
  
  uint V = pow(2,D);  
  uvec shift = zeros<uvec>(V);
  bvec b;
  for(uint d = 0; d < D; ++d){
    b = binmask(d,D);
    shift(find(b == 1)) += coef[d];
  }
  return shift;
}

bool coords_in_bound(const umat & coords,
		      const uvec & lens,
		      const Mask & oob){

  uint D = coords.n_cols;
  for(uint d = 0; d < D; d++){
    if(any(coords.col(d) >= lens(d)))
      return false;
  }
  return all(all(coords.rows(oob.pos) == OOB_COORD));
}

uvec coords_to_indices(const umat & coords,
		       const uvec & lens,
		       const Mask & oob){
  // Converts a matrix of coordinates to indices

  assert(coords_in_bound(coords,lens,oob));
  
  uvec coef = c_order_coef(lens);
  uvec idx = uvec(coords.n_rows);
  idx.rows(oob.neg) = coords.rows(oob.neg) * coef;
  idx.rows(oob.pos).fill(oob_index(lens));
  
  return idx;
}

umat indices_to_coords(const uvec & idx, const uvec & lens){
  uvec coef = c_order_coef(lens);
  
  uint D = lens.n_elem;
  uint N = idx.n_elem;
  
  umat crd = umat(N,D);
  uvec curr_idx = idx;
  for(uint d = 0; d < D; ++d){
    crd.col(d) =  floor(curr_idx / coef(d));
    curr_idx = vec_mod(curr_idx, coef(d));
  }

  // Deal with oob index
  uvec oob = find(idx >= prod(lens));
  for(uvec::const_iterator it = oob.begin();
      it != oob.end(); ++it){
    crd.row(*it) = conv_to<urowvec>::from(lens);
  }
  
  return crd;
}

umat least_coord(const mat & points,
		 const RegGrid & grid,
		 const Mask & oob_mask){
  /*
    Returns the actual coordinate of the least cutpoint in
    the hypercube that the point is in.
   */
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(N > 0);
  assert(check_dim(grid,D));

  rowvec low = grid.low.t();
  urowvec num = grid.num_cells.t();
  rowvec span = grid.high.t() - grid.low.t();
  
  mat scaled = mat(N,D);
  if(oob_mask.n_neg > 0){
    scaled.rows(oob_mask.neg) =
      num % (points.each_row(oob_mask.neg) - low) / span;
  }
  if(oob_mask.n_pos > 0){
    scaled.rows(oob_mask.pos).fill(OOB_COORD);
  }

  umat cuts = conv_to<umat>::from(floor(scaled));
  return cuts;
}

void out_of_bounds(Mask & oob_mask,
		   const mat & points,
		   const RegGrid & grid){
  uint N = points.n_rows;
  uint D = points.n_cols;
  bvec mask = zeros<bvec>(N);

  mat T = (points.each_row() - grid.low.t());
  mask(find(any(T < 0,1))).fill(1);

  T = (grid.high.t() - points.each_row());
  mask(find(any(T < GRID_FUDGE,1))).fill(1);

  generate_mask(oob_mask,mask);
}

sp_mat point_to_idx_dist(const mat & points,
			  const RegGrid & grid){
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(check_dim(grid,D));

  // Get the width of the rectangles
  rowvec delta = conv_to<rowvec>::from(width(grid));

  // Get the oob points; deal with seperately
  // Assumption: there aren't enough of these to make handling them
  // cleverly worth it.
  Mask oob_mask;  
  out_of_bounds(oob_mask,points,grid);

  // Get the least vertex for the cell enclosing each point
  umat low_coords = least_coord(points,grid,oob_mask);
  cout << "low:\n" << low_coords << endl;
  
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
  uvec low_indices = coords_to_indices(low_coords,grid.num_cells,oob_mask);
  
  // Calculate the shift for the regular grid cells
  uvec shift = cell_shift_coef(grid.num_cells);

  // Build the sparse matrix
  rowvec data = vectorise(W,1);
  umat loc = umat(2,N*V); // Location pairs (i,j)

  // Fill in loc matrix
  uint oob_idx = oob_index(grid);
  uint I = 0;
  cout << "OOB:" << oob_idx << endl;
  for(uint j = 0; j < V; ++j){
    for(uint i = 0; i < N; ++i){
      uint row = low_indices(i) + shift(j);
      loc(0,I) = min(row,oob_idx);
      loc(1,I) = i; // Column: point index
      ++I;
    }
  }
  
  sp_mat dist = sp_mat(loc,data);
  return dist;
}
  

int main(int argc, char** argv)
{
  uint R = 1; // Repetitions (for timing)
  uint N = 2; // Grid resolution (Number of cut
  uint D = 2; // Dimension

  RegGrid g;
  g.low = zeros<vec>(D);
  g.high = N*ones<vec>(D);
  g.num_cells = N*ones<uvec>(D); // Number of CELLS

  mat GP = make_points(g);
  cout << "Grid points:\n" << GP << endl;
  
  mat P = mat("2.1 1.1");

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  for(uint i = 0; i < R; ++i){
    cout << "P:\n" << P << endl;
    sp_mat D = point_to_idx_dist(P,g);
    cout << "Dist:\n" << D << endl;
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  //cout << duration_cast<microseconds>( t2 - t1 ).count();
  
  return 0;
}
