#include "hallway.h"
#include "misc.h"
#include "io.h"
#include "lcp.h"
#include "solver.h"

int main(int argc, char** argv)
{
  uint N = 25;
  vec points = linspace<vec>(0,1,N+1).head(N);

  double p_stick = 0.25;
  double gamma = 0.99;
  LCP lcp = build_hallway_lcp(N,p_stick,gamma);

  KojimaSolver solver = KojimaSolver();
  SolverResult sol = solver.aug_solve(lcp);

  Archiver arch = Archiver();
  arch.add_vec("p",sol.p);
  arch.add_vec("d",sol.d);
  arch.add_sp_mat("M",lcp.M);
  arch.write("test.sol");
}
