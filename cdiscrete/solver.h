#ifndef __Z_SOLVER_INCLUDED__
#define __Z_SOLVER_INCLUDED__

#include "lcp.h"
#include <armadillo>

using namespace std;
using namespace arma;

double max_steplen(const vec & x,
                   const vec & dx);
double steplen_heuristic(const vec & x,
                         const vec & y,
                         const vec & dx,
                         const vec & dy,
                         double scale);
double sigma_heuristic(double sigma,
                       double steplen);

typedef pair<vec,vec> pd_pair; // primal/dual pair

class KojimaSolver{
 public:
  KojimaSolver();
  void solve(const LCP & lcp,vec & x, vec & y) const;
  pd_pair aug_solve(const LCP & lcp) const;
 protected:
  double comp_thresh;
  uint max_iter;
  bool verbose;
};

class ProjectiveSolver{
 public:
  ProjectiveSolver();
  void solve(const PLCP & plcp,vec& x, vec& y, vec& w) const;
  pd_pair aug_solve(const PLCP & plcp) const;
  
 protected:
  double comp_thresh;
  uint max_iter;
  bool verbose;
};

#endif
