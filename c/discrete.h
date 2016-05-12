#ifndef __DISCRETE_INCLUDED__
#define __DISCRETE_INCLUDED__

#include <armadillo>
using namespace std;
using namespace arma;

//=====================================
// STRUCTURES

// Regular grid structure
struct RegGrid {
vec low;
vec high;
uvec num; // Number of CELLS per dimension; will +1 cutpoints
};

//=====================================
// SIMPLE EXTENSIONS TO ARMADILLO
bool check_dim(const RegGrid & grid, uint D);

// Element-wise modulus
uvec vec_mod(const uvec & a, uint n);

// Replace elements index in cnd by scalar
void replace(vec & v, double val,const uvec & cnd);
void replace(uvec & v, uint val,const uvec & cnd);

// Like above, but for a particular column
// NB: this is to avoid passing A.col(i), which is a special subview type
void replace_col(umat & M, uint col, double val,const uvec & cnd);

// Binary ops
uvec num2binvec(uint n,uint D);
uvec binmask(uint d, uint D);

//=======================================
// MAKE POINTS
// Enumerates the points of a D-dimension grid from a D-long vector of cuts
//NB: 'C' style ordering: last column changes most rapidly
mat make_points(const vector<vec> grids);

// INDEXING
// Coefficients for converting D-tuples into indicies
uvec c_order_coef(const uvec & lens);
uvec cell_shift_coef(const uvec & lens);

// Coverts D-coordinates into indicies
uvec coord_to_index(const umat & crd, const uvec & lens);

// Coverts indicies into D-coordinates
umat index_to_coord(const uvec & idx, const uvec & lens);

// Returns the coordinates of smallest grid-point of the hypercube
// that each point is in
umat least_coord(const mat & points, const RegGrid & grid);

//==========================================
// MAIN DISCRETIZATION FUNCTION
// Returns a sparse matrix that encodes the multi-linear interpolation
// coefficients for every point.
// e.g. If given an NxD matrix of points, returns a GxN matrix
// where at most 2**D elements of each column is non-zero
// and G is the number of mesh point (+1 for oob)
sp_mat point_to_idx_dist(const mat & points,const RegGrid & grid);
  
  
#endif
