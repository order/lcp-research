#ifndef __Z_MISC_INCLUDED__
#define __Z_MISC_INCLUDED__

#include <armadillo>
#include <random>
#include <chrono>
#include <vector>

#define ALMOST_ZERO 1e-15
#define PRETTY_SMALL 1e-9

double time_delta(const clock_t & start);

void print_shape(const arma::uvec & u);
void print_shape(const arma::vec & v);
void print_shape(const arma::mat & A);

// Make points from a grid
template <typename M, typename V> M make_points(const std::vector<V> & grids);

arma::SizeMat uvec2sizemat(const arma::uvec & pair);
arma::SizeCube uvec2sizecube(const arma::uvec & cube);

//=====================================
// BOOLEAN
typedef arma::Col<unsigned char> bvec; 

//=====================================
// SIMPLE EXTENSIONS TO ARMADILLO

// Element-wise modulus

arma::uvec vec_mod(const arma::uvec & a, uint n);
arma::vec vec_mod(const arma::vec & a, double n);

arma::umat divmod(const arma::uvec & a, uint n);

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

arma::vec lp_norm(const arma::mat & A,double p,uint dir);
arma::mat pdist(const arma::mat & A); // 
arma::vec dist(const arma::mat & A, const arma::mat & B);
arma::vec dist(const arma::mat & A, const arma::rowvec & b);

double rectify(double x);
arma::vec rectify(const arma::vec & x);
double soft_threshold(double x, double thresh);
arma::vec soft_threshold(arma::vec x, double thresh);

// Return B(i,j) = min(A(i,j),b(j))
arma::mat row_min(const arma::mat & A, const arma::rowvec & b);
arma::mat row_max(const arma::mat & A, const arma::rowvec & b);
void row_min_inplace(arma::mat & A, const arma::rowvec & b);
void row_max_inplace(arma::mat & A, const arma::rowvec & b);
void min_inplace(arma::vec & u, const arma::vec & b);
void max_inplace(arma::vec & u, const arma::vec & b);

void scalar_min_inplace(arma::mat & A, double s);
void scalar_max_inplace(arma::mat & A, double s);

// Index of maximum column for each row in V
uint argmax(const arma::vec & v);
uint argmin(const arma::vec & v);
uint argmax(const arma::uvec & v);
uint argmin(const arma::uvec & v);
arma::uvec col_argmax(const arma::mat & V);
arma::uvec col_argmin(const arma::mat & V);

double quantile(const arma::vec & v, double q);

// Like above, but for a particular column
// NB: this is to avoid passing A.col(i), which is a special subview type
void replace_col(arma::umat & M, uint col, double val,const arma::uvec & cnd);

// Binary ops
bvec num2binvec(uint n,uint D);
bvec binmask(uint d, uint D);

// Some shifting stuff
template <typename D> D last(const arma::Col<D> & v);
template <typename D> arma::Col<D> rshift(const arma::Col<D> & v);
// Non-circular shift


//Sparse stuff
typedef std::vector<std::vector<arma::sp_mat> > block_sp_mat;
typedef std::vector<arma::sp_mat> block_sp_vec;

block_sp_vec block_lmult(const arma::sp_mat & A, const block_sp_vec & Bs);
block_sp_vec block_rmult(const arma::sp_mat & A, const block_sp_vec & Bs);

arma::sp_mat block_mat(const block_sp_mat & B);
arma::sp_mat block_diag(const block_sp_vec & D);
arma::sp_mat spdiag(const arma::vec & v, int d);


arma::sp_mat sp_submatrix(const arma::sp_mat & A,
                          const arma::uvec & row_idx,
                          const arma::uvec & col_idx);

block_sp_mat sp_partition(const arma::sp_mat & A,
                          const arma::uvec & idx_1,
                          const arma::uvec & idx_2);

arma::sp_mat sp_normalise(const arma::sp_mat & A,uint p,uint d);

arma::sp_mat h_join_sp_mat_vector(const std::vector<arma::sp_mat> & comps);

double sparsity(const arma::sp_mat & A);

#endif
