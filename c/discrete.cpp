#include <iostream>
#include <assert.h>
#include "discrete.h"

#include <chrono>


using namespace std;
using namespace std::chrono;
using namespace arma;

bool check_dim(const RegGrid & grid, uint D){
  // Make sure all dimensions of a RegGrid are consistant
  return grid.low.n_elem == D
    && grid.high.n_elem == D
    && grid.num.n_elem == D;
}

uvec vec_mod(const uvec & a, uint n){
  return a - (a / n) * n;
}

void replace(vec & v, double val, const uvec & cnd){
  v(cnd) = val * ones<vec>(cnd.n_elem);  
}

void replace(uvec & v, uint val, const uvec & cnd){
  v(cnd) = val * ones<uvec>(cnd.n_elem);  
}

void replace_col(umat & M, uint c, uint val, const uvec & cnd){
  uvec v = M.col(c);
  v(cnd) = val * ones<uvec>(cnd.n_elem);  
}

uvec num2binvec(uint n,uint D){
  // Convert n to a D-bit binary vector
  // Little endian
  assert(n < pow(2,D));
  uvec b = uvec(D);
  for(uint d = 0; d < D; ++d){
    b(d) = ((1 << d) & n) >> d;
  }
  return b;
}

uvec binmask(uint d, uint D){
  // For all of numbers in 0,..., 2**D-1, do they have bit d lit up?
  // e.g. binmask(0,D) will be: [0,1,0,1,...]
  // e.g. binmask(1,D) will be: [0,0,1,1,0,0,1,1,...]

  uint N = pow(2,D);
  uvec mask = uvec(N);
  for(uint b = 0; b < N; ++b){
    mask[b] = ((1 << d) & b) >> d; // Shift 1 over, mask, then shift back.
  }
  return mask;
}

mat make_points(const vector<vec> grids)
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
  uvec b;
  for(uint d = 0; d < D; ++d){
    b = binmask(d,D);
    shift(find(b == 1)) += coef[d];
  }
  return shift;
}

uvec coord_to_index(const umat & crd, const uvec & lens){
  // Converts a matrix of coordinates to indices

  for(uint c = 0; c < crd.n_cols; ++c){
    assert(not any(crd.col(c) > lens(c)));
    assert(not any(crd.col(c) < 0));
  }
  
  uvec coef = c_order_coef(lens);
  uvec idx = crd * coef;
  
  return idx;
}

umat index_to_coord(const uvec & idx, const uvec & lens){
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
	      const RegGrid & grid){
  /*
    Returns the actual coordinate of the least cutpoint in
    the hypercube that the point is in.
   */
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(check_dim(grid,D));

  umat cuts = umat(N,D);
  for(uint d = 0; d < D; ++d){
    uint n = grid.num(d);
    double l = grid.low(d);
    double h = grid.high(d);

    vec t = floor(n * (points.col(d) - l) / (h - l));
    cuts.col(d) = conv_to<uvec>::from(t);

    // Fuzz to convert [l,h) to [l,h]
    uvec fuzz = find(points.col(d) >= h
		     && points.col(d) < h+1e-12);
    replace_col(cuts, d, n-1, fuzz);

    // OOB
    uvec oob = find(points.col(d) < l
		    || points.col(d) >= h+1e-12);
    replace_col(cuts, d, n, oob);
  }
  return cuts;
}

sp_mat point_to_idx_dist(const mat & points,
			  const RegGrid & grid){
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(check_dim(grid,D));

  uint num_grid_points = prod(grid.num)+1; // +1 for oob
  umat low_coords = least_coord(points,grid); // least vertex in cube
  rowvec delta = (grid.high.t() - grid.low.t()) / grid.num.t(); //grid increment
  /*
    Want to get  distance from least cell cut points (denoted by the @ below):
      o-----o
     / |   /|
    o--+--o |
    |  o--|-o
    | /   |/ 
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
  uint V = pow(2,D);
  mat W = ones<mat>(N,V);
  uint halfV = pow(2,D-1);

  for(uint d = 0; d < D; ++d){
    // Iterate through the dimensions
    uvec mask = binmask(d,D);
    // Update half with the distance from the low side of the cube
    W.cols(find(mask == 1)) %=
      repmat(norm_diff.col(d),1,halfV);
    // Update the rest with the distance from the far side
    W.cols(find(mask == 0)) %=
      repmat(1.0 - norm_diff.col(d),1,halfV);
  }


  // Calculate the index of the low grid point
  uvec low_indices = coord_to_index(low_coords,grid.num);
  
  // Calculate the shift for the regular grid cells
  uvec shift = cell_shift_coef(grid.num);

  // Build the sparse matrix
  rowvec data = vectorise(W,1);
  umat loc = umat(2,N*V); // Location pairs (i,j)

  // Fill in loc matrix
  uint I = 0;
  for(uint j = 0; j < V; ++j){
    for(uint i = 0; i < N; ++i){
      uint row = low_indices(i) + shift(j);
      loc(0,I) = min(row,num_grid_points);
      loc(1,I) = i; // Column: point index
      ++I;
    }
  }

  cout << loc << endl;
  
  sp_mat dist = sp_mat(loc,data);
  return dist;
}
  

int main(int argc, char** argv)
{
  uint R = 1;
  uint N = 4;
  uint D = 2;
  
  vec x = vec(N);
  for(uint i = 0; i < N; ++i){x[i] = float(i);}
  vector<vec> grids;
  for(uint d = 0; d < D; ++d){grids.push_back(x);}

  RegGrid g;
  g.low = zeros<vec>(D);
  g.high = N*ones<vec>(D);
  g.num = N*ones<uvec>(D); // Number of CELLS
  
  //mat P = make_points(grids);
  //mat P = randu<mat>(1,D);
  mat P = mat("0.1 0.1");

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  for(uint i = 0; i < R; ++i){
    sp_mat D = point_to_idx_dist(P,g);
    cout << D << endl;
  }
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  //cout << duration_cast<microseconds>( t2 - t1 ).count();
  
  return 0;
}
