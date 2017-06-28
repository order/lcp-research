#ifndef __Z_MISC_INCLUDED__
#define __Z_MISC_INCLUDED__

#include <armadillo>
#include <random>
#include <chrono>
#include <vector>

#define ALMOST_ZERO 1e-15
#define PRETTY_SMALL 1e-9

using namespace arma;

void print_shape(const uvec & u);
void print_shape(const vec & v);
void print_shape(const mat & A);

// Make points from a grid
template <typename M, typename V> M make_points(const std::vector<V> & grids);

SizeMat uvec2sizemat(const uvec & pair);
SizeCube uvec2sizecube(const uvec & cube);

//=====================================
// BOOLEAN
typedef Col<unsigned char> bvec; // boolean vector; bool doesn't work in arma

//=====================================
// SIMPLE EXTENSIONS TO ARMADILLO

// Element-wise modulus

uvec vec_mod(const uvec & a, uint n);
vec vec_mod(const vec & a, double n);

umat divmod(const uvec & a, uint n);

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

vec lp_norm(const mat & A,double p,uint dir);
mat pdist(const mat & A); // 
vec dist(const mat & A, const mat & B);
vec dist(const mat & A, const rowvec & b);

double rectify(double x);
vec rectify(const vec & x);
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

double quantile(const vec & v, double q);

// Like above, but for a particular column
// NB: this is to avoid passing A.col(i), which is a special subview type
void replace_col(umat & M, uint col, double val,const uvec & cnd);

// Binary ops
bvec num2binvec(uint n,uint D);
bvec binmask(uint d, uint D);

// Some shifting stuff
template <typename D> D last(const Col<D> & v);
template <typename D> Col<D> rshift(const Col<D> & v);
// Non-circular shift


//Sparse stuff
typedef std::vector<std::vector<sp_mat> > block_sp_mat;
typedef std::vector<sp_mat> block_sp_vec;

block_sp_vec block_lmult(const sp_mat & A, const block_sp_vec & Bs);
block_sp_vec block_rmult(const sp_mat & A, const block_sp_vec & Bs);

sp_mat block_mat(const block_sp_mat & B);
sp_mat block_diag(const block_sp_vec & D);
sp_mat spdiag(const vec & v);
sp_mat spdiag(const vec & v,const int d);


sp_mat sp_submatrix(const sp_mat & A,
                          const uvec & row_idx,
                          const uvec & col_idx);

block_sp_mat sp_partition(const sp_mat & A,
                          const uvec & idx_1,
                          const uvec & idx_2);

sp_mat sp_normalise(const sp_mat & A,uint p,uint d);

double sparsity(const sp_mat & A);

#endif
