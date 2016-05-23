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

// Return B(i,j) = min(A(i,j),b(j))
mat row_min(const mat & A, const rowvec & b);
mat row_max(const mat & A, const rowvec & b);
void row_min_inplace(mat & A, const rowvec & b);
void row_max_inplace(mat & A, const rowvec & b);

// Index of maximum column for each row in V
uvec col_argmax(const mat & V);
uvec col_argmin(const mat & V);

// Like above, but for a particular column
// NB: this is to avoid passing A.col(i), which is a special subview type
void replace_col(umat & M, uint col, double val,const uvec & cnd);

// Binary ops
bvec num2binvec(uint n,uint D);
bvec binmask(uint d, uint D);

#endif
