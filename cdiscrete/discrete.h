#ifndef __Z_DISCRETE_INCLUDED__
#define __Z_DISCRETE_INCLUDED__

#include <armadillo>
#include <climits>
#include "misc.h"

using namespace std;
using namespace arma;

#define OOB_COORD UINT_MAX // OOB "coordinate"
#define OOB_STATE DOUBLE_MAX
#define GRID_FUDGE 1e-12

//=====================================
// STRUCTURES
// Aliases for rows and col dims.
uint num_actions(const mat & actions);
uint action_dim(const mat & actions);

uint num_states(const mat & states);
uint state_dim(const mat & states);

// Regular grid structure
struct RegGrid {
vec low;
vec high;
uvec num_cells; // Number of CELLS per dimension;
// Number of grid cuts per dimension is +1 this
};

void print_grid(const RegGrid & grid);

uint grid_dim(const RegGrid & grid);
uvec num_grid_points_per_dim(const RegGrid & grid);
uint num_grid_points(const RegGrid & grid);
uint oob_index(const RegGrid & grid);
uint oob_index(const uvec & points_per_dim);
vec width(const RegGrid & grid);
bool verify(const RegGrid & grid);
bool check_dim(const RegGrid & grid, uint D);

//=======================================
// MAKE POINTS
// Enumerates the points of a D-dimension grid from a D-long vector of cuts
//NB: 'C' style ordering: last column changes most rapidly
vector<vec> vectorize(const RegGrid & grid);
mat make_points(const RegGrid & grid);
mat make_points(const vector<vec> & grids);

//=======================================
// INDEXING
// Coefficients for converting D-tuples into indicies
uvec c_order_coef(const uvec & points_per_dim);
uvec cell_shift_coef(const uvec & points_per_dim);

// Coverts D-coordinates into indicies
bool coords_in_bound(const umat & coords,
		     const uvec & points_per_dim,
		     const Mask & oob);
uvec coords_to_indices(const umat & coords,
		       const uvec & points_per_dim,
		       const Mask & oob);

// Coverts indicies into D-coordinates
// Can either supply the out-of-bound binary vector, or have it generated
umat indices_to_coords(const uvec & indices,
		       const uvec & points_per_dim);
umat indices_to_coords(const uvec & indices,
		       const uvec & points_per_dim,
		       const bvec & oob_mask);

// Returns the coordinates of smallest grid-point of the hypercube
// that each point is in
umat least_coords(const mat & points,const RegGrid & grid, const Mask & oob_mask);

// Mask of oob points (e.g. row indices)
void out_of_bounds(Mask & oob_mask,
		    const mat & points,
		    const RegGrid & grid);

//==========================================
// MAIN DISCRETIZATION FUNCTION
// Returns a sparse matrix that encodes the multi-linear interpolation
// coefficients for every point.
// e.g. If given an NxD matrix of points, returns a GxN matrix
// where at most 2**D elements of each column is non-zero
// and G is the number of mesh point (+1 for oob)
sp_mat point_to_idx_dist(const mat & points,const RegGrid & grid);


//===========================================
// INTERPOLATE
vec interp_fn(const vec & val,
	      const mat & points,
	      const RegGrid & grid);
mat interp_fns(const mat & vals,
	       const mat & points,
	       const RegGrid & grid);

uvec max_interp_fns(const mat & vals,
		    const mat & points,
		    const RegGrid & grid);
uvec min_interp_fns(const mat & vals,
		    const mat & points,
		    const RegGrid & grid);

//===========================================
// PYTHON STUFF
void print_list(uvec L);

#endif
