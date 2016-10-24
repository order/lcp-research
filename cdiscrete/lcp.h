#ifndef __Z_LCP_INCLUDED__
#define __Z_LCP_INCLUDED__

#include "simulator.h"
#include <armadillo>
#include <assert.h>

using namespace arma;

struct LCP{
  LCP();
  LCP(const sp_mat &,
      const vec &);
  LCP(const sp_mat &,
      const vec &,
      const bvec &);  
  sp_mat M;
  vec q;
  bvec free_vars;
  void write(const string &);
};

struct PLCP{
  PLCP();
  PLCP(const sp_mat &,
       const sp_mat &,
       const vec &);
  PLCP(const sp_mat &,
       const sp_mat &,
       const vec &,
       const bvec &);
  sp_mat P;
  sp_mat U;
  vec q;
  bvec free_vars;
  void write(const string &);
};

vector<sp_mat> build_E_blocks(const Simulator * sim,
                              const Discretizer * disc,
                              double gamma,
                              bool include_oob);

// Build skew-symmetric matrix out of blocks.
// [0     E1 E2]
// [-E1.T 0  0 ]
// [-E2.T 0  0 ]
sp_mat build_M(const vector<sp_mat> & E_blocks);

vec build_q_vec(const Simulator * sim,
            const Discretizer * disc,
            double gamma,
            bool include_oob);
LCP build_lcp(const Simulator * sim,
              const Discretizer * disc,
              double gamma,
              bool include_oob = false,
              bool value_nonneg = true);

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
