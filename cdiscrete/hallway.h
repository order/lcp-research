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

sp_mat build_smoothed_identity(uint N,double p);
LCP build_smoothed_hallway_lcp(const uint N,
                               const double p_stick,
                               const double p_smooth,
                               const double gamma);
#endif
