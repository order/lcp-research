#ifndef __Z_HALLWAY_INCLUDED__
#define __Z_HALLWAY_INCLUDED__


#include <armadillo>
#include "lcp.h"

using namespace arma;
using namespace std;


sp_mat build_hallway_P(const uint N,
                       const double p_stick,
                       const int action);
sp_mat build_hallway_M(const uint N,
                       const double p_stick,
                       const double gamma);

vec build_hallway_q(const uint N);

LCP build_hallway_lcp(const uint N,
                      const double p_stick,
                      const double gamma);

#endif
