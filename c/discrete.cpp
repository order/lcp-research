#include <iostream>
#include <armadillo>
#include <assert.h>
#include <set>

using namespace std;
using namespace arma;

struct RegGrid {
  vec low;
  vec high;
  uvec num;
};

inline bool check_dim(const RegGrid & grid, uint D){
  return grid.low.n_elem == D
    && grid.high.n_elem == D
    && grid.num.n_elem == D;
}

template <typename V>
V nans(uint n){
  V v = V(n);
  v.fill(datum::nan);
  return v;
}

uvec vec_mod(const uvec & a, const uvec n){
  assert(a.n_elem == n.n_elem);
  return a - (a / n) % n; // '%' overloaded to be elem-mult
}

uvec vec_mod(const uvec & a, uint n){
  return a - (a / n) * n;
}

template <typename V>
mat div_by_vec(const mat & A, const V & x){
  // Divides the columns of A by elements of x
  // So B[:,i] = A[:,i] / x[i]
  assert(A.n_cols == x.n_elem);
  mat B = mat(A.n_rows, A.n_cols);
  for(uint d = 0; d < x.n_elem; ++d){
    B.col(d) /= x(d);
  }
  return B;
}

template <typename V>
mat add_by_vec(const mat & A, const V & x){
  //Adds elements of x to columns of A
  // So B[:,i] = A[:,i] + x[i]
  assert(A.n_cols == x.n_elem);
  mat B = mat(A.n_rows, A.n_cols);
  for(uint d = 0; d < x.n_elem; ++d){
    B.col(d) += x(d);
  }
  return B;
}

template <typename V>
mat mult_by_vec(const mat & A, const V & x){
  //Adds elements of x to columns of A
  // So B[:,i] = A[:,i] + x[i]
  assert(A.n_cols == x.n_elem);
  mat B = mat(A.n_rows, A.n_cols);
  for(uint d = 0; d < x.n_elem; ++d){
    B.col(d) *= x(d);
  }
  return B;
}

template <typename V>
void replace(V & v,double rpl,const uvec & cnd){
  for(uvec::const_iterator it = cnd.begin();
      it != cnd.end(); ++it){
    v(*it) = rpl;
  }
}

template <typename M>
void replace_col(M & A,uint c,double rpl,const uvec & cnd){
  for(uvec::const_iterator it = cnd.begin();
      it != cnd.end(); ++it){
    A.col(c)(*it) = rpl;
  }
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
  uint n = lens.n_elem;
  uvec coef = uvec(n);

  uint agg = 1;
  for(uint i = n; i > 0; --i){
    coef(i-1) = agg;
    agg *= lens(i-1);
  }
  return coef;
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
    replace_col<umat>(cuts,d, n-1, fuzz);

    // OOB
    uvec oob = find(points.col(d) < l
		    || points.col(d) >= h+1e-12);
    replace_col<umat>(cuts,d, n, oob);
  }
  return cuts;
}

void point_to_idx_dist(const mat & points,
			 const RegGrid & grid){
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(check_dim(grid,D));

  // Grid coords (as mat)
  umat coords = least_coord(points,grid);
  uvec I = coord_to_index(coords,grid.num);

  vec delta = (grid.high - grid.low) / grid.num;

  assert(D == grid.low.n_elem);

  cout << size(coords * delta) << endl;
  cout << size(grid.low) << endl;

  /*
    Want to get  distance from least cell cut points:
    
    o-----o
    |  P  |
    | /   |
    L-----o

    This distance is normalized by cell size (delta vector)

    P / d - (C + (l /d)) = (P - (l + C*d)) / d

   */
  mat norm_points = div_by_vec(points,delta); // mat / vec
  vec norm_low = grid.low / delta; // vec / vec
  mat norm_least_pos = add_by_vec(conv_to<mat>::from(coords),
    norm_low); // mat + vec
  mat norm_diff = norm_points - norm_least_pos; // mat - mat

  mat W = mat(N,pow(2,D));
  for(uint b = 0; b < pow(2,D); ++b){
  cout << b << ',' << conv_to<urowvec>::from(num2binvec(b,D))
       << endl;
  }
}
  

int main(int argc, char** argv)
{  
  mat P = randu<mat>(10,2);
  
  cout << "Points:" << endl<< P << endl;

  RegGrid grid;
  grid.low = zeros<vec>(2);
  grid.high = ones<vec>(2);
  grid.num = 4*ones<uvec>(2);

  point_to_idx_dist(P,grid);
  
  return 0;
}
