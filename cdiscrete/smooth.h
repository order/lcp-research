#ifndef __Z_SMOOTH_INCLUDED__
#define __Z_SMOOTH_INCLUDED__

#include <armadillo>

using namespace std;
using namespace arma;

sp_mat gaussian_smoother(const Points & points,
                         const double bandwidth,
                         const double zero_thresh);

#endif
