#ifndef __Z_SOLVER_INCLUDED__
#define __Z_SOLVER_INCLUDED__

#include "lcp.h"
#include <armadillo>

#include <Eigen/Sparse>

#define GH_SYSTEM_FILEPATH "/home/epz/scratch/test_gh_system.arch"



double max_steplen(const arma::vec & x,
                   const arma::vec & dx);
double steplen_heuristic(const arma::vec & x,
                         const arma::vec & y,
                         const arma::vec & dx,
                         const arma::vec & dy,
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
  SolverResult solve(const LCP & lcp, arma::vec & x, arma::vec & y) const;

  double comp_thresh;
  uint max_iter;
  bool verbose;
  bool iter_verbose; // Just print iteration
  bool save_system;
  double regularizer;
  double aug_rel_scale;
  double initial_sigma;
};

class ProjectiveSolver{
 public:
  ProjectiveSolver();
  SolverResult aug_solve(const PLCP & plcp) const;
  SolverResult aug_solve(const PLCP & plcp,
			 arma::vec & x, arma::vec & y) const;

  SolverResult solve(const PLCP & plcp,
		     arma::vec & x, arma::vec & y, arma::vec & w) const;

  double comp_thresh;
  uint max_iter;
  bool verbose;
  bool iter_verbose; // Just print iteration
  double regularizer;
  double aug_rel_scale;
  double initial_sigma;
};

class ValueIteration{
 public:
  ValueIteration();
  vec solve(const vector<sp_mat> & p_blocks, double gamma, const mat & costs);

  double change_thresh;
  uint max_iter;
  bool verbose;
};

#endif
