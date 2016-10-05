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

struct SolverResult{
  vec p;
  vec d;
  uint iter;
  SolverResult();
  SolverResult(const vec & p, const vec & d, uint iter);
  void trim_final();
  void write(const string & filename) const;
};

class KojimaSolver{
 public:
  KojimaSolver();
  SolverResult aug_solve(const LCP & lcp) const;
  SolverResult solve(const LCP & lcp, vec & x, vec & y) const;

  double comp_thresh;
  uint max_iter;
  bool verbose;
  double regularizer;
  double aug_rel_scale;
  double initial_sigma;
};

class ProjectiveSolver{
 public:
  ProjectiveSolver();
  SolverResult aug_solve(const PLCP & plcp) const;
  SolverResult aug_solve(const PLCP & plcp,vec & x,vec & y) const;

  SolverResult solve(const PLCP & plcp,vec& x, vec& y, vec& w) const;

  double comp_thresh;
  uint max_iter;
  bool verbose;
  double regularizer;
  double aug_rel_scale;
  double initial_sigma;
};

#endif
