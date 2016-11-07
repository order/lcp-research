#include "hallway.h"
#include "misc.h"
#include "io.h"
#include "lcp.h"
#include "solver.h"
#include "basis.h"

sp_mat make_value_basis(const vec & points){
  uint N = points.n_elem;

  uint k = 15;
  uint K = k+1;

  vec centers = linspace(0,1,k);
  
  double bw = 60;
  mat basis = mat(N,K);
  uint I = 0;
  basis.col(I++) = ones<vec>(N);
  for(uint w = 0; w < k; w++){
    basis.col(I++) = gaussian(points,centers(w),bw);
  }
  assert(I == K);
  return sp_mat(orth(basis));
}

mat perturb_q(const mat & Q, const double noise_std){
  uint N = Q.n_rows;
  uint A = Q.n_cols;

  mat tildeQ = mat(size(Q));
  tildeQ.col(0) = Q.col(0); // Same weights
  for(uint a = 1; a < A; a++){
    tildeQ.col(a) = Q.col(a) + noise_std * randn<vec>(N);
  }
  return tildeQ;
}

vec sinc(const vec & x){
  vec s = vec(size(x));
  uvec mask = find(x != 0);
  s(mask) = sin(2.0*datum::pi*x(mask)) / x(mask);
  mask = find(x == 0);
  s(mask).fill(1);
  return s;
}

int main(int argc, char** argv)
{

  arma_rng::set_seed_random();
  // Set up the 1D space
  uint N = 512;
  uint A = 3;
  vec points = linspace<vec>(0,1,N+1).head(N);

  // Build the LCP
  double p_stick = 0.25;
  double p_smooth = 0.55;
  double gamma = 0.99;

  // Build the reference blocks
  vector<sp_mat> blocks = build_hallway_blocks(N,p_stick,gamma);

  // Smooth out the blocks with a convolver
  vec x = linspace(-1,1,9);
  //vec g = exp(-5*abs(x));
  vec g = exp(-5*x%x);
  //vec g = sinc(3*x);
  g /= accu(abs(g));
  cout << "Smoothing vector: " << g.t() << endl;
  sp_mat smoother = build_convolution_matrix(N,g);

  // Build and pertrub the q
  mat Q = build_hallway_q(N);
  mat tildeQ = perturb_q(Q,0.25);

  // TODO: perturb the operator
  bvec free_vars = zeros<bvec>((A+1)*N);
  free_vars.head(N).fill(1);

  LCP rlcp = LCP(build_M(blocks),vectorise(Q),free_vars);
  LCP lcp = LCP(build_M(blocks),vectorise(tildeQ),free_vars);
  LCP slcp = smooth_lcp(smoother,blocks,tildeQ,free_vars);

  sp_mat value_basis = make_value_basis(points);
  PLCP plcp = approx_lcp(value_basis,speye(N,N),blocks,tildeQ,free_vars);
  PLCP splcp = approx_lcp(value_basis,smoother,blocks,tildeQ,free_vars);

  // Solve the problem
  KojimaSolver solver = KojimaSolver();
  solver.verbose = false;
  ProjectiveSolver psolver = ProjectiveSolver();
  psolver.verbose = false;
  SolverResult rsol = solver.aug_solve(rlcp);
  SolverResult sol = solver.aug_solve(lcp);
  SolverResult ssol = solver.aug_solve(slcp);
  SolverResult psol = psolver.aug_solve(plcp);
  SolverResult spsol = psolver.aug_solve(splcp);

  // Build the PLCP problem

  // Record the solution and problem data
  Archiver arch = Archiver();
  arch.add_vec("ref_p",rsol.p);
  arch.add_vec("ref_d",rsol.d);
  arch.add_vec("p",sol.p);
  arch.add_vec("d",sol.d);
  arch.add_vec("sp",ssol.p);
  arch.add_vec("sd",ssol.d);
  arch.add_vec("pp",psol.p);
  arch.add_vec("pd",psol.d);
  arch.add_vec("spp",spsol.p);
  arch.add_vec("spd",spsol.d);
  arch.write("test.sol");
}
