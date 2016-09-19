#ifndef __Z_LCP_INCLUDED__
#define __Z_LCP_INCLUDED__

#include "simulator.h"
#include <armadillo>

using namespace arma;

struct LCP{
  LCP(const sp_mat &,
      const vec &);
  sp_mat M;
  vec q;
  void write(const string &);
};

vector<sp_mat> build_E_blocks(const Simulator * sim,
                              const Discretizer * disc,
                              double gamma,
                              bool include_oob);
sp_mat build_M(const vector<sp_mat> & E_blocks);

vec build_q(const Simulator * sim,
            const Discretizer * disc,
            double gamma,
            bool include_oob);
LCP build_lcp(const Simulator * sim,
              const Discretizer * disc,
              double gamma,
              bool include_oob);
#endif
