#include <iostream>
#include <assert.h>
#include "misc.h"
#include <wordexp.h>

#include <vector>

#include <boost/math/special_functions/sign.hpp>

using namespace std;
using namespace arma;

void print_shape(const uvec & u){
  cout << '(' << u.n_elem << ')' << endl;
}

void print_shape(const vec & v){
  cout << '(' << v.n_elem << ')' << endl;
}

void print_shape(const mat & A){
  cout << '(' << A.n_rows
	    << ',' << A.n_cols << ')' << endl;
}

template<typename M, typename V> M make_points(const vector<V> & grids)
{
  // Makes a mesh out of the D vectors
  // 'C' style ordering... last column changes most rapidly
  
  // Figure out the dimensions of things
  uint D = grids.size();
  uint N = 1;
  for(auto const & it : grids){
    N *= it.size();
  }
  M P = M(N,D); // Create the matrix
  
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
template mat make_points(const vector<vec> & grids);
template umat make_points(const vector<uvec> & grids);
template imat make_points(const vector<ivec> & grids);


SizeMat uvec2sizemat(const uvec & pair){
  assert(2 == pair.n_elem);
  return size(pair(0), pair(1));
}
SizeCube uvec2sizecube(const uvec & cube){
  assert(3 == cube.n_elem);
  return size(cube(0), cube(1), cube(2)); 
}

//=================================================

uvec vec_mod(const uvec & a, uint n){
  return a - (a / n) * n;
}

vec vec_mod(const vec & a, double n){
  return a - floor(a / n) * n;
}

umat divmod(const uvec & a, uint n){
  /*
   * Returns two columns:
   * 1) Multiplier floor(a / n)
   * 2) Mod (a % n)
   */
  umat ret = umat(a.n_elem, 2);
  ret.col(0) = floor(a/n);
  ret.col(1) = a - ret.col(0) * n;
  return ret;
}


template<typename V> bvec in_interval(const V & x,
				      double lb,
				      double ub){
  uint N = x.n_elem;
  bvec ret = ones<bvec>(N);
  ret(find(x < lb)).fill(0);
  ret(find(x > ub)).fill(0);
  return ret;
}

template<typename M,typename V>
  bvec in_intervals(const M & A,
		    const V & lb,
		    const V & ub){
  uint N = A.n_rows;
  uint D = A.n_cols;
  bvec ret = ones<bvec>(N);
  for(uint d = 0; d < D; d++){
    ret(find(A.col(d) < lb(d))).fill(0);
    ret(find(A.col(d) > ub(d))).fill(0);
  }
  return ret;
}
template bvec in_intervals<mat,vec>(const mat & A,
				    const vec & lb,
				    const vec & ub);

template<typename V> bool is_logical(const V & v){
  vec u = unique(conv_to<vec>::from(v));
  if(u.n_elem > 2) return false;

  if(u.n_elem == 2){
    return (u(0)==0) and (u(1) == 1);
  }

  assert(u.n_elem == 1);
  return (u(0) == 0) or (u(0) == 1);
}

template<typename V>
bvec land(const V & a, const V & b){
  assert(is_logical(a));
  assert(is_logical(b));
  return min(a,b);
}

template<typename V>
bvec lor(const V & a, const V & b){
  assert(is_logical(a));
  assert(is_logical(b));
  return max(a,b);
}

template<typename V>
bvec lnot(const V & v){
  assert(is_logical(v));
  return 1 - v;
}
template bvec lnot(const bvec & v);

template <typename M, typename V> M row_mult(const M & A,
					     const V & b){
  return A.each_row() % b;
}
template mat row_mult(const mat & A, const rowvec & b);


template <typename M, typename V> M row_diff(const M & A,
					     const V & b){
  return A.each_row() - b;
}
template mat row_diff(const mat & A, const rowvec & b);

template <typename M, typename V> M row_add(const M & A,
					    const V & b){
  return A.each_row() + b;
}
template mat row_add(const mat & A, const rowvec & b);

template <typename M, typename V> M row_divide(const M & A,
					       const V & b){
  return A.each_row() / b;
}
template mat row_divide(const mat & A, const rowvec & b);

vec lp_norm(const mat & A,double p,uint dir){
  /*
    Does either the column-wise (dir=0) or row-wise (dir=1) p-norm
  */
  return pow(sum(pow(A,p),dir),1.0 / p);
}

mat pdist(const mat& A){
  uint N = A.n_rows;
  mat D = mat(N,N);
  
  for(uint i = 0; i < N; i++){
    // Subview nonsense preventing easily doing efficient thing
    vec d = lp_norm(A.each_row() - A.row(i),2,1);
    D.col(i) = d;
  }
  return D;
}

vec dist(const mat & A, const mat & B){
  return lp_norm(A - B,2,1);
}

vec dist(const mat & A, const rowvec & b){
  return lp_norm(A.each_row() - b,2,1);
}

double dist(const vec & v, const vec & u){
  return norm(v - u);
};

double rectify(double x){
  return max(0.0,x);
}

vec rectify(const vec & x){
  return max(zeros<vec>(x.n_elem),x);
}

double soft_threshold(double x, double thresh){
  return boost::math::sign(x) * max(0.0,abs(x) - thresh);
}

vec soft_threshold(vec x, double thresh){
  vec t = abs(x) - thresh;
  t(find(t < 0)).fill(0);
  return sign(x) % t;
}

mat row_min(const mat & A, const rowvec & b){
  uint D = A.n_cols;
  mat B = A;
  uvec col_idx = uvec(1);  
  for(uint d = 0; d < D; d++){
    col_idx.fill(d);
    B.submat(find(B.col(d) > b(d)),col_idx).fill(b(d));
  }
  return B;
}
mat row_max(const mat & A, const rowvec & b){
  uint D = A.n_cols;
  mat B = A;
  uvec col_idx = uvec(1);  
  for(uint d = 0; d < D; d++){
    col_idx.fill(d);
    B.submat(find(B.col(d) < b(d)),col_idx).fill(b(d));
  }
  return B;
}

void row_min_inplace(mat & A, const rowvec & b){
  uint D = A.n_cols;
  uvec col_idx = uvec(1);  
  for(uint d = 0; d < D; d++){
    col_idx.fill(d);
    A.submat(find(A.col(d) > b(d)),col_idx).fill(b(d));
  }
}
void row_max_inplace(mat & A, const rowvec & b){
  uint D = A.n_cols;
  uvec col_idx = uvec(1);
  for(uint d = 0; d < D; d++){
    col_idx.fill(d);
    A.submat(find(A.col(d) < b(d)),col_idx).fill(b(d));
  }
}

void min_inplace(vec & u, const vec & b){
  uvec idx = find(u > b);
  u(idx) = b(idx);
}
void max_inplace(vec & u, const vec & b){
  uvec idx = find(u < b);
  u(idx) = b(idx);
}

void scalar_min_inplace(mat & A, double s){
  uvec idx = find(A > s);
  A(idx).fill(s);
}

void scalar_max_inplace(mat & A, double s){
  uvec idx = find(A < s);
  A(idx).fill(s);
}

uint argmax(const vec & v){
  assert(v.n_elem > 0);
  uvec res = find(v == max(v),1);
  return res(0);
}

uint argmin(const vec & v){
  return argmax(-v);
}

uint argmax(const uvec & v){
  return argmax(conv_to<vec>::from(v));
}

uint argmin(const uvec & v){
  return argmax(-conv_to<vec>::from(v));
}

uvec col_argmax(const mat & V){
  uint N = V.n_rows;
  vec m = max(V,1);
  assert(N == m.n_elem);

  uvec am = uvec(N);
  uvec::fixed<1> idx;
  for(uint i = 0; i < N; i++){
    idx = find(V.row(i) == m(i), 1);
    assert(1 == idx.n_elem);
    am(i) = idx(0); // find first index that matches
  }
  return am;
}

uvec col_argmin(const mat & V){
  return col_argmax(-V);
}

double quantile(const vec & v, double q){
  assert(0.0 <= q);
  assert(1.0 >= q);
  // Sort the vector
  vec s = sort(v);

  // Get index associated with
  // the quantile
  uint N = v.n_elem;
  double p = (double) N * q;
  int i = (uint)floor(p);
  // Interpolation weights
  double theta = (p - i);
  assert(0 <= theta);
  assert(1 >= theta); 
  
  if(i == N-1){
    assert(theta < ALMOST_ZERO);
    return s(i);
  } 

  // Interpolate
  return (1 - theta)*s(i) + theta * s(i+1);
}


//=================================================
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

template <typename D> D last(const Col<D> & v){
  return (v.tail(1))(0);
}

template <typename D>
Col<D> rshift(const Col<D> & v){
  Col<D> u = shift(v,1);
  assert(u(1) == v(0));
  u(0) = 0;
  return u;
}

block_sp_vec block_lmult(const sp_mat & A,
                        const block_sp_vec & Bs){

  uint n = Bs.size();
  block_sp_vec Cs;
  Cs.reserve(n);
  for(uint i = 0; i < n; i++){
    Cs.push_back(A * Bs[i]);
  }
  return Cs;
}

block_sp_vec block_rmult(const sp_mat & A,
                         const block_sp_vec & Bs){
  
  uint n = Bs.size();
  block_sp_vec Cs;
  Cs.reserve(n);
  for(uint i = 0; i < n; i++){
    Cs.push_back(Bs[i]*A);
  }
  return Cs;
}


sp_mat block_mat(const block_sp_mat & B){
  uint b_rows = B.size();
  assert(b_rows > 0);
  uint b_cols = B[0].size();
  assert(b_cols > 0);

  uvec rows = zeros<uvec>(b_rows);  
  uvec cols = zeros<uvec>(b_cols);
  uint nnz = 0;

  // Gather block size information
  for(uint i = 0; i < b_rows; i++){
    assert(b_cols == B[i].size());
    for(uint j = 0; j < b_cols; j++){
      if(rows[i]>0 and max(rows[i],B[i][j].n_rows) != rows[i]){
	cerr << "[BMAT ERROR] Incompatible row dimensions" << endl;
	exit(1);
      }
      if(cols[j]>0 and max(cols[i],B[i][j].n_cols) != cols[i]){
	cerr << "[BMAT ERROR] Incompatible col dimensions" << endl;
	exit(1);
      }
      rows[i] = max(rows[i],B[i][j].n_rows);
      cols[j] = max(cols[j],B[i][j].n_cols);
      nnz += B[i][j].n_nonzero;
    }
  }

  uvec cum_rows = cumsum(rows);
  uvec cum_cols = cumsum(cols);
  uvec row_off = rshift(cum_rows);
  uvec col_off = rshift(cum_cols);
  uint R = last(cum_rows);
  uint C = last(cum_cols);

  vec values = vec(nnz); // template
  umat loc = umat(2,nnz);
  
  typedef sp_mat::const_iterator sp_iter;
  uint I = 0;
  for(uint j = 0; j < b_cols; j++){
    for(uint i = 0; i < b_rows; i++){
      for(sp_iter it = B[i][j].begin();
	  it != B[i][j].end(); ++it){	
	values(I) = (*it);
	loc(0,I) = it.row() + row_off(i);
	loc(1,I) = it.col() + col_off(j);
	++I;
      }
    }
  }
  assert(nnz == I);
  
  sp_mat ret =  sp_mat(loc,values,R,C);
  return ret;
}

sp_mat block_diag(const block_sp_vec & D){
  uint d = D.size();
  block_sp_mat blocks;
  blocks.resize(d);
  for(uint i = 0; i < d; i++){
    blocks[i].resize(d);
    blocks[i][i] = D[i];
  }
  return block_mat(blocks);
}

sp_mat spdiag(const vec & v){
  return spdiag(v,0);
}

sp_mat spdiag(const vec & v, const int d){
  uint n = v.n_elem;
  uint N = n + abs(d);

  umat loc = umat(2,n);
  uint r_start = max(0,-d);
  uint c_start = max(0,d);
  loc.row(0) = regspace<urowvec>(r_start,r_start+n-1);
  loc.row(1) = regspace<urowvec>(c_start,c_start+n-1);
  
  return sp_mat(loc,v,N,N);
}

sp_mat sp_submatrix(const sp_mat & A,
                          const uvec & row_idx,
                          const uvec & col_idx){
  // Apparently non-contiguous sub-matrixing is not supported yet
  assert(row_idx.is_sorted());
  assert(col_idx.is_sorted());
  
  sp_mat ret = sp_mat(row_idx.n_elem,
                      col_idx.n_elem);

  for(sp_mat::const_iterator it = A.begin();
      it !=  A.end(); ++it){

    uvec I = find(row_idx == it.row());
    if(0 == I.n_elem){
      continue; // Not in this submatrix
    }
    uvec J = find(col_idx == it.col());
    if(0 == J.n_elem){
      continue; // Not in this submatrix
    }
    assert(1 == I.n_elem);
    assert(1 == J.n_elem);
    ret(I(0),J(0)) = *it;
  }
  return ret;
}

block_sp_mat sp_partition(const sp_mat & A,
                          const uvec & idx_1,
                          const uvec & idx_2){
  block_sp_mat ret = block_sp_mat(2);
  ret[0].resize(2);
  ret[1].resize(2);

  vector<uvec> idxs = {idx_1,idx_2};
  for(uint i = 0; i < 2; i++){
    for(uint j = 0; j < 2; j++){
      ret[i][j] = sp_submatrix(A,idxs[i],idxs[j]);
    }
  }
  return ret;
}

sp_mat sp_normalise(const sp_mat & A,
                    uint p,uint d){
  if(1 == d)
    return sp_normalise(A.t(),p,0);
  
  assert(0 == d); // Column-wise
  sp_mat B = sp_mat(size(A));
  
  uint C = A.n_cols;
  for(uint c = 0; c < C; c++){
    // determine column norm
    double z = 0;
    for(sp_mat::const_iterator it = A.begin_col(c);
        it != A.end_col(c); ++it){
      z += pow(*it,p);   
    }
    assert(z > 0);
    z =  pow(z,1.0 / (double) p);
    B.col(c) = A.col(c) / z;
  }
  return B;
}

double sparsity(const sp_mat & A){
  return (double)A.n_nonzero / (double) A.n_elem;
}
