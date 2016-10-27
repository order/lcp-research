#ifndef __Z_HALLWAY_INCLUDED__
#define __Z_HALLWAY_INCLUDED__


#include <armadillo>
#include "lcp.h"

using namespace arma;
using namespace std;


sp_mat build_hallway_P(const uint N,
                       const double p_stick,
                       const int action);

vector<sp_mat> build_hallway_blocks(const uint N,
                                    const double p_stick,
                                    const double gamma);

mat build_hallway_q(const uint N);

LCP build_hallway_lcp(const uint N,
                      const double p_stick,
                      const double gamma);

sp_mat build_convolution_matrix(uint N, const vec & v);

vector<sp_mat> make_freebie_flow_bases(const sp_mat & value_basis,
                                       const vector<sp_mat> blocks);

PLCP approx_lcp(const vec & points,
                const sp_mat & value_basis,
                const block_sp_vec & blocks,
                const mat & Q,
                const bvec & free_vars);
#endif
