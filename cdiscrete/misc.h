#ifndef __Z_MISC_INCLUDED__
#define __Z_MISC_INCLUDED__

#include <armadillo>
#include <random>
#include <chrono>

//=====================================
// RANDOM GENERATOR
extern unsigned SEED;
extern std::mt19937 MT_GEN;

using namespace arma;

void print_shape(const uvec & u);
void print_shape(const vec & v);
void print_shape(const mat & A);

//=====================================
// BOOLEAN
typedef Col<unsigned char> bvec; // boolean vector; bool doesn't work in arma

//=====================================
// SIMPLE EXTENSIONS TO ARMADILLO

// Element-wise modulus
uvec vec_mod(const uvec & a, uint n);

// Interval
template<typename V> bvec in_interval(const V & x,
				   double lb,
				   double ub);
template<typename M,typename V>
  bvec in_intervals(const M & A,
		    const V & lb,
		    const V & ub);

// Logical operations
template<typename V> bool is_logical(const V &);


template<typename V> bvec land(const V &, const V &);
template<typename V> bvec lor(const V &, const V &);
template<typename V> bvec lnot(const V &);

// Row-wise operation
template <typename M, typename V> M row_mult(const M & A, const V & b);
template <typename M, typename V> M row_diff(const M & A, const V & b);
template <typename M, typename V> M row_add(const M & A, const V & b);
template <typename M, typename V> M row_divide(const M & A, const V & b);

vec row_2_norm(const mat & A);
vec dist(const mat & A, const mat & B);
vec dist(const mat & A, const rowvec & b);

double rectify(double x);
vec rectify(vec & x);
double soft_threshold(double x, double thresh);
vec soft_threshold(vec x, double thresh);

// Return B(i,j) = min(A(i,j),b(j))
mat row_min(const mat & A, const rowvec & b);
mat row_max(const mat & A, const rowvec & b);
void row_min_inplace(mat & A, const rowvec & b);
void row_max_inplace(mat & A, const rowvec & b);
void min_inplace(vec & u, const vec & b);
void max_inplace(vec & u, const vec & b);

// Index of maximum column for each row in V
uint argmax(const vec & v);
uint argmin(const vec & v);
uint argmax(const uvec & v);
uint argmin(const uvec & v);
uvec col_argmax(const mat & V);
uvec col_argmin(const mat & V);

// Like above, but for a particular column
// NB: this is to avoid passing A.col(i), which is a special subview type
void replace_col(umat & M, uint col, double val,const uvec & cnd);

// Binary ops
bvec num2binvec(uint n,uint D);
bvec binmask(uint d, uint D);

#endif
