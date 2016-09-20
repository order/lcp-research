#include <iostream>
#include <assert.h>
#include "misc.h"
#include <wordexp.h>

#include <boost/math/special_functions/sign.hpp>


using namespace arma;

unsigned SEED = std::chrono::system_clock::now().time_since_epoch().count();
std::mt19937 MT_GEN = std::mt19937(SEED);


void print_shape(const uvec & u){
  std::cout << '(' << u.n_elem << ')' << std::endl;
}

void print_shape(const vec & v){
  std::cout << '(' << v.n_elem << ')' << std::endl;
}

void print_shape(const mat & A){
  std::cout << '(' << A.n_rows
	    << ',' << A.n_cols << ')' << std::endl;
}

mat make_points(const std::vector<vec> & grids)
{
  // Makes a mesh out of the D vectors
  // 'C' style ordering... last column changes most rapidly
  
  // Figure out the dimensions of things
  uint D = grids.size();
  uint N = 1;
  for(std::vector<vec>::const_iterator it = grids.begin();
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

//=================================================

uvec vec_mod(const uvec & a, uint n){
  return a - (a / n) * n;
}

vec vec_mod(const vec & a, double n){
  return a - floor(a / n) * n;
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
  return pow(sum(pow(A,p),dir),1.0 / p);
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
  return std::max(0.0,x);
}

vec rectify(vec & x){
  vec t = x;
  t(find(t < 0.0)).fill(0.0);
  return t;
}

double soft_threshold(double x, double thresh){
  return boost::math::sign(x) * std::max(0.0,std::abs(x) - thresh);
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

sp_mat bmat(const block_sp_mat & B){

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
      if(rows[i]>0 and std::max(rows[i],B[i][j].n_rows) != rows[i]){
	cerr << "[BMAT ERROR] Incompatible row dimensions" << endl;
	exit(1);
      }
      if(cols[j]>0 and std::max(cols[i],B[i][j].n_cols) != cols[i]){
	cerr << "[BMAT ERROR] Incompatible col dimensions" << endl;
	exit(1);
      }
      rows[i] = std::max(rows[i],B[i][j].n_rows);
      cols[j] = std::max(cols[j],B[i][j].n_cols);
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
