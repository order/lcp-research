#include "hallway.h"
#include "misc.h"
#include "io.h"
#include "lcp.h"
#include "solver.h"
#include "basis.h"

sp_mat make_value_basis(const vec & points){
  uint N = points.n_elem;

  uint k = 15;
  uint K = k+2;

  vec centers = linspace(0,1,k);
  
  double bw = 60;
  mat basis = mat(N,K);
  uint I = 0;
  basis.col(I++) = ones<vec>(N);
  basis.col(I) = zeros<vec>(N);
  basis(N/2,I++) = 1;
  for(uint w = 0; w < k; w++){
    basis.col(I++) = gaussian(points,centers(w),bw);
  }
  assert(I == K);
  return sp_mat(orth(basis));
}

vector<sp_mat> make_flow_bases(const sp_mat & value_basis,
                               const vector<sp_mat> p_blocks){
  // TODO: think more carefully about q
  vector<sp_mat> flow_bases;
  uint A = p_blocks.size();
  for(uint a = 0; a < A; a++){
    sp_mat raw_basis = p_blocks.at(a) * value_basis;
    flow_bases.push_back(sp_mat(orth(mat(raw_basis)))); // Orthonorm
  }
  return flow_bases;
}


// Should be pretty generic; put in a more general file
PLCP approx_hallway_lcp(const vec & points,
                        const LCP & lcp,
                        const vector<sp_mat> & p_blocks){
  /* Build the approximate PLCP based on:
     1) An MDP LCP specificied via the P blocks
     2) A value bases (make_value_basis)
  */
  
  uint N = points.n_elem;
  uint A = p_blocks.size();
  cout << "(N,A): (" << N << "," << A << ")" << endl;
  // Make the value bases
  sp_mat value_basis = make_value_basis(points);
  cout << "Value basis size: " << size(value_basis) << endl;
    
  // Build the flow bases
  // This is the "freebie basis" for the flow based on the
  // Smoothed LCP
  vector<sp_mat> flow_bases = make_flow_bases(value_basis,
                                              p_blocks);
  assert(A == flow_bases.size());
  
  block_sp_vec b_blocks;
  b_blocks.reserve(A + 1);
  b_blocks.push_back(value_basis);
  b_blocks.insert(b_blocks.end(),
                  flow_bases.begin(),
                  flow_bases.end());
  assert((A+1) == b_blocks.size());
  sp_mat P = block_diag(b_blocks);
  cout << "P: " << size(P) << endl;

  // Build the U vector based on 
  uint V = lcp.q.n_elem;
  cout << "q: " << size(lcp.q) << endl;
  cout << "M: " << size(lcp.M) << endl;
  sp_mat M = lcp.M + 1e-10 * speye(V,V); // Regularize
  sp_mat U = P.t() * M * P * P.t();
  vec q = P *(P.t() * lcp.q);

  bvec free_vars = zeros<bvec>(V); // TODO
  PLCP plcp = PLCP(P,U,q,free_vars);

  return plcp;
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
  double p_smooth = 0.99;
  double gamma = 0.997;

  LCP lcp = build_hallway_lcp(N,p_stick,gamma);
  vector<sp_mat> blocks = build_hallway_blocks(N,p_stick,gamma);
  assert(A == blocks.size());
  
  LCP slcp = build_smoothed_hallway_lcp(N,p_stick,p_smooth,gamma);
  vector<sp_mat> sblocks;
  sp_mat smoother = build_smoothed_identity(N,p_smooth);
  for(uint a = 0; a< A; a++)
    sblocks.push_back(smoother * blocks[a]);
  
  PLCP plcp = approx_hallway_lcp(points,lcp,blocks);
  PLCP splcp = approx_hallway_lcp(points,slcp,sblocks);


  // Solve the problem
  KojimaSolver solver = KojimaSolver();
  solver.verbose = false;
  ProjectiveSolver psolver = ProjectiveSolver();
  SolverResult sol = solver.aug_solve(lcp);
  SolverResult ssol = solver.aug_solve(slcp);
  SolverResult psol = psolver.aug_solve(plcp);
  SolverResult spsol = psolver.aug_solve(splcp);

  // Build the PLCP problem

  // Record the solution and problem data
  Archiver arch = Archiver();
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
