#ifndef __Z_LCP_INCLUDED__
#define __Z_LCP_INCLUDED__

#include "simulator.h"
#include <armadillo>
#include <assert.h>

using namespace arma;

struct LCP{
  LCP(const sp_mat &,
      const vec &);
  LCP();
  sp_mat M;
  vec q;
  void write(const string &);
};

struct PLCP{
  PLCP();
  PLCP(const sp_mat &,
       const sp_mat &,
       const vec &);
  sp_mat P;
  sp_mat U;
  vec q;
  void write(const string &);
};

vector<sp_mat> build_E_blocks(const Simulator * sim,
                              const Discretizer * disc,
                              double gamma,
                              bool include_oob);
sp_mat build_M(const vector<sp_mat> & E_blocks);

vec build_q_vec(const Simulator * sim,
            const Discretizer * disc,
            double gamma,
            bool include_oob);
LCP build_lcp(const Simulator * sim,
              const Discretizer * disc,
              double gamma,
              bool include_oob);

// Augmented LCP
// Use this LCP if feasible start isn't known.
LCP augment_lcp(const LCP & original,
                vec & x,
                vec & y,
                double rel_scale=10.0);
PLCP augment_plcp(const PLCP & original,
                  vec & x,
                  vec & y,
                  vec & w,
                  double scale);
#endif
