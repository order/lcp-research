#ifndef __ARMA_MISC_INCLUDED__
#define __ARMA_MISC_INCLUDED__

#include <armadillo>

using namespace arma;

void print_shape(const uvec & u);
void print_shape(const vec & v);
void print_shape(const mat & A);

//=====================================
// BOOLEAN
typedef Col<unsigned char> bvec; // boolean vector; bool doesn't work in arma
// Mask structure
struct Mask {
  bvec mask;
  uvec pos; // positive occurances
  uvec neg; // negative occurances
  uint n_pos;
  uint n_neg;
};
void generate_mask(Mask & mask, uvec & source);
void generate_mask(Mask & mask, bvec & source);

//=====================================
// SIMPLE EXTENSIONS TO ARMADILLO

// Element-wise modulus
uvec vec_mod(const uvec & a, uint n);

// Row-wise operation
mat row_mult(const mat & A, const rowvec & b);
mat row_diff(const mat & A, const rowvec & b);
mat row_add(const mat & A, const rowvec & b);

vec row_2_norm(const mat & A);
vec dist(const mat & A, const mat & B);
vec dist(const mat & A, const rowvec & b);

mat row_min(const mat & A, const rowvec & b);
mat row_max(const mat & A, const rowvec & b);
void row_min_inplace(mat & A, const rowvec & b);
void row_max_inplace(mat & A, const rowvec & b);

// Like above, but for a particular column
// NB: this is to avoid passing A.col(i), which is a special subview type
void replace_col(umat & M, uint col, double val,const uvec & cnd);

// Binary ops
bvec num2binvec(uint n,uint D);
bvec binmask(uint d, uint D);

#endif
