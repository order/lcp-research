#include <iostream>
#include <assert.h>
#include "misc.h"

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

//=================================================

uvec vec_mod(const uvec & a, uint n){
  return a - (a / n) * n;
}

template<typename V> bvec in_interval(const V & x,
				      double lb,
				      double ub){
  uint N = x.n_elem;
  bvec ret = zeros<bvec>(N);
  ret(x >= lb).fill(1);
  ret(x <= ub).fill(1);
  return ret;
}

template<typename M,typename V>
  bvec in_intervals(const M & A,
		    const V & lb,
		    const V & ub){
  uint N = A.n_rows;
  uint D = A.n_cols;
  bvec ret = zeros<bvec>(N);
  for(uint d = 0; d < D; d++){
    ret(A.col(d) >= lb(d)).fill(1);
    ret(A.col(d) <= ub(d)).fill(1);
  }
  return ret;
}

template<typename V> bool is_logical(const V & v){
  vec u = unique(v);
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

mat row_mult(const mat & A, const rowvec & b){
  return A.each_row() % b;
}
mat row_diff(const mat & A, const rowvec & b){
  return A.each_row() - b;
}
mat row_add(const mat & A, const rowvec & b){
  return A.each_row() + b;
}

vec row_2_norm(const mat & A){
  return sqrt(sum(square(A),1));
}

vec dist(const mat & A, const mat & B){
  return row_2_norm(A - B);
}

vec dist(const mat & A, const rowvec & b){
  return row_2_norm(A.each_row() - b);
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
