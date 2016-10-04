#include <vector>
#include <assert.h>

#include "solver.h"
#include "io.h"

double max_steplen(const vec & x,
                   const vec & dx){
  if(all(dx >= 0))
    return 1.0; // Everything moving away from boundary
  double step = min(- x(find(dx < 0))
                    /dx(find(dx < 0)));
  assert(step > 0);
  return step;
}

double steplen_heuristic(const vec & x,
                         const vec & y,
                         const vec & dx,
                         const vec & dy,
                         double scale){
  double x_step = max_steplen(x,dx);
  double y_step = max_steplen(y,dy);
  
  return min(vec{1.0,
        scale*x_step,
        scale*y_step});
  
}
double sigma_heuristic(double sigma,
                       double steplen){
  double max_sigma = 0.999;
  double min_sigma = 0.1;
  if(steplen >= 0.8)
    sigma *= 0.975;
  else if(steplen < 0.2)
      sigma = 0.75 + 0.25*sigma;
  else if(steplen < 1e-3)
    sigma = max_sigma;
  return max(min_sigma,min(max_sigma,sigma));
}

SolverResult::SolverResult(){}
SolverResult::SolverResult(const vec & ap,
                           const vec & ad,
                           uint aiter) : p(ap),d(ad),iter(aiter){}

void SolverResult::trim_final(){
  uint N = p.n_elem;
  p.resize(N-1);
  d.resize(N-1);
}

void SolverResult::write(const string & filename) const{
  Archiver arch;
  arch.add_vec("p",p);
  arch.add_vec("d",d);
  arch.write(filename);
}
////////////////////////////////////////////////////////
// Kojima

KojimaSolver::KojimaSolver(){
  comp_thresh = 1e-8;
  max_iter = 500;
  verbose = true;
  regularizer = 1e-8;
  aug_rel_scale = 0.75;
}

// Really should be always using this.
SolverResult KojimaSolver::aug_solve(const LCP & lcp) const{
  uint N = lcp.q.n_elem;
  double scale = aug_rel_scale * (double) N;
  // Balance feasibility and complementarity
  vec x,y;
  LCP alcp = augment_lcp(lcp,x,y,scale);
  SolverResult sol = solve(alcp,x,y);
  if(verbose)
    cout << "Final augmented variable value: " << sol.p.tail(1) << endl;
  sol.trim_final();
  return sol;
}

SolverResult KojimaSolver::solve(const LCP & lcp,
                                 vec & x,
                                 vec & y) const{
  superlu_opts opts;
  opts.equilibrate = true;
  opts.permutation = superlu_opts::COLAMD;
  opts.refine = superlu_opts::REF_SINGLE;
  
  vec q = lcp.q;
  sp_mat M = lcp.M + regularizer * speye(size(lcp.M));
  uint N = q.n_elem;
  cout << "Initial residual: " << norm(M*x + q - y) << endl;

  assert(N == x.n_elem);
  assert(N == y.n_elem);

  // Figure out what is free (-inf,inf)
  // and what is bound to be non-negative [0,inf)
  bvec free_vars = lcp.free_vars;
  uvec bound_idx = find(0 == free_vars);
  uvec free_idx = find(1 == free_vars);
  assert(N == bound_idx.n_elem + free_idx.n_elem);
  uint NB = bound_idx.n_elem; // number of bound vars
  uint NF = free_idx.n_elem; // number of free vars
  
  /*
    In what follows, the primal variables x are partitioned into
    free variables and bound variables x = [f;b]'
    Likewise, the dual variables are partitioned into y = [0,s]'
   */
  
  /* The Newton system:
  [M_ff  M_fb  0][df]   [M_f x + q_f]
  [M_bf  M_bb -I][db] + [M_b x + q_b - s]
  [0     S     B][dv]   [u1 + VBe]
  Where "M_ff" is the free-free block
  Overwrite the S,B diagonals every iteration*/

  // Split M matrix into blocks based on free and bound indicies
  block_sp_mat M_part = sp_partition(M,free_idx,bound_idx);
  sp_mat M_recon = block_mat(M_part);
  assert(PRETTY_SMALL > norm(M_recon - M));
  vec qf = q(free_idx);
  vec qb = q(bound_idx);
  vec b = x(bound_idx);
  vec f = x(free_idx);
  vec s = y(bound_idx);

  // Build the Newton matrix
  vector<vector<sp_mat>> block_G;
  block_G.push_back(block_sp_vec{sp_mat(),sp_mat(),sp_mat(NB,NB)});
  block_G.push_back(block_sp_vec{-M_part[0][0],-M_part[0][1],sp_mat()});
  block_G.push_back(block_sp_vec{-M_part[1][0],-M_part[1][1],speye(NB,NB)});
 
  // Start iteration
  double mean_comp, steplen, sigma = 0.95;
  uint iter;  
  for(iter = 0; iter < max_iter; iter++){
    if(verbose)
      cout << "---Iteration " << iter << "---" << endl;
    assert(all(0 == y(free_idx)));

    // Mean complementarity
    mean_comp = dot(b,s) / (double) NB;
    if(mean_comp < comp_thresh)
      break;

    block_G[0][1] = spdiag(s);
    block_G[0][2] = spdiag(b);
    sp_mat G = block_mat(block_G);
    assert(size(N + NB,N + NB) == size(G));    
                    
    // Form RHS from residual and complementarity
    vec h = vec(N + NB);
    vec res_f = M_part[0][0]*f + M_part[0][1]*b + qf;
    vec res_b = M_part[1][0]*f + M_part[1][1]*b + qb - s;
    h.head(NB) = sigma * mean_comp - b % s;
    h.subvec(NB,size(res_f)) = res_f;
    h.tail(NB) = res_b;
    
    //Archiver arch;
    //arch.add_sp_mat("G",G);
    //arch.add_vec("h",h);
    //arch.write("test.sys");

    // Solve and extract directions
    vec dir = spsolve(G,h,"superlu",opts);
    assert((N+NB) == dir.n_elem);
    vec df = dir.head(NF);
    vec db = dir.subvec(NF,N-1);
    assert(NB == db.n_elem);
    vec ds = dir.tail(NB);    
    vec dir_recon = join_vert(df,join_vert(db,ds));
    assert(PRETTY_SMALL > norm(dir_recon-dir));

    steplen = steplen_heuristic(b,s,db,ds,0.9);
    sigma = sigma_heuristic(sigma,steplen);

    f += steplen * df;
    b += steplen * db;
    s += steplen * ds;

    if(verbose){
      double res = norm(res_f) + norm(res_b);
      cout <<"\t Mean complementarity: " << mean_comp
           <<"\n\t Residual norm: " << res
           <<"\n\t |df|: " << norm(df)
           <<"\n\t |db|: " << norm(db)
           <<"\n\t |ds|: " << norm(ds)
           <<"\n\t Step length: " << steplen
           <<"\n\t Centering sigma: " << sigma << endl;
    }
  }
  if(verbose){
    cout << "Finished"
         <<"\n\t Final mean complementarity: " << mean_comp << endl;
  }
  x(free_idx) = f;
  x(bound_idx) = b;
  y(free_idx).fill(0);
  y(bound_idx) = s;
  return SolverResult(x,y,iter);
}


/////////////////////////////////////////////////////////////
// PROJECTIVE

ProjectiveSolver::ProjectiveSolver(){
  comp_thresh = 1e-8;
  max_iter = 500;
  verbose = true;
  regularizer = 1e-8;
  aug_rel_scale = 0.75;

}

// Really should be always using this.


SolverResult ProjectiveSolver::aug_solve(const PLCP & plcp) const{
  uint N = plcp.P.n_rows;
  uint K = plcp.P.n_cols;
  double scale = aug_rel_scale * (double) N;
  // Balance feasibility and complementarity
  vec x,y,w;
  PLCP aplcp = augment_plcp(plcp,x,y,w,scale);
  SolverResult sol = solve(aplcp,x,y,w);
  sol.trim_final();
  return sol;
}

SolverResult ProjectiveSolver::aug_solve(const PLCP & plcp,
                                         vec & x,
                                         vec & y ) const{
  uint N = plcp.P.n_rows;
  uint K = plcp.P.n_cols;
  double scale = 0.5 * (double) N; // Balance feasibility and complementarity
  vec w;
  PLCP aplcp = augment_plcp(plcp,x,y,w,scale);
  return solve(aplcp,x,y,w);
}

SolverResult ProjectiveSolver::solve(const PLCP & plcp,
                            vec & x,
                            vec & y,
                            vec & w) const{
  sp_mat P = plcp.P;
  sp_mat U = plcp.U;
  vec q = plcp.q;
  uint N = P.n_rows;
  uint K = P.n_cols;
  assert(size(P.t()) == size(U));

  bvec free_vars = plcp.free_vars;
  uvec bound_idx = find(0 == free_vars);
  uvec free_idx = find(1 == free_vars);
  assert(N == bound_idx.n_elem + free_idx.n_elem);
  uint NB = bound_idx.n_elem; // number of bound vars
  uint NF = free_idx.n_elem; // number of free vars
  assert(NF == accu(conv_to<uvec>::from(free_vars)));

  if(verbose)
    cout << "Free variables: \t" << NF << endl
         << "Non-neg variables:\t" << NB << endl;
  sp_mat J = join_vert(sp_mat(NF,NB),speye(NB,NB));
  assert(size(N,NB) == size(J));
  assert(PRETTY_SMALL > norm(y(free_idx)));

  if(verbose)
    cout << "Forming pre-computed products..." << endl;
  mat PtP = mat(P.t() * P);
  assert(size(K,K) == size(PtP));
  mat PtPU = PtP*U;
  assert(size(K,N) == size(PtPU));
  mat Pt_PtPU = P.t() - PtPU;
  assert(size(K,N) == size(Pt_PtPU));
  mat PtPUP = PtPU * P;
  assert(size(K,K) == size(PtPUP));

  vec Ptq = P.t() * q;
  if(verbose)
    cout << "Done..." << endl;
  
  assert(N == x.n_elem);
  assert(N == y.n_elem);
  assert(K == w.n_elem);
  
  double steplen, sigma;
  sigma = 0.95;
  
  superlu_opts slu_opts;
  slu_opts.equilibrate = true; // This is important for conditioning
  slu_opts.permutation = superlu_opts::COLAMD;
  slu_opts.refine = superlu_opts::REF_SINGLE;

  uint iter;
  double mean_comp;
  for(iter = 0; iter < max_iter; iter++){    
    if(verbose)
      cout << "---Iteration " << iter << "---" << endl;
    // Mean complementarity
    vec s = y(bound_idx);
    vec b = x(bound_idx);
    mean_comp = dot(s,b) / (double) NB;
    if(mean_comp < comp_thresh)
      break;

    // Generate reduced Netwon system
    mat C = s+b;
    vec g = sigma * mean_comp - s % b;
    assert(NB == g.n_elem);

    // NB: A,G,and h have opposite sign from python version    
    mat A = Pt_PtPU * J * spdiag(1.0 / C);
    assert(size(K,NB) == size(A));
     
    mat G = (A * spdiag(s)) * J.t() * P - PtPUP;
    assert(size(K,K) == size(G));

    vec Ptr = P.t() * (J *  s) - PtPU*x - Ptq;
    vec h = Ptr + A*g;
    assert(K == h.n_elem);

    // Options don't make much difference
    vec dw = arma::solve(G,h,solve_opts::equilibrate);
    assert(K == dw.n_elem);

    // Recover dy
    vec JPdw = J * P * dw;
    assert(NB == JPdw.n_elem);
    vec ds = (g - s % JPdw) / C;
    assert(NB == ds.n_elem);
    
    // Recover dx
    vec dx = (J * ds) + (P * dw);
    assert(N == dx.n_elem);    

    steplen = steplen_heuristic(x(bound_idx),s,dx(bound_idx),ds,0.9);
    sigma = sigma_heuristic(sigma,steplen);

    x += steplen * dx;
    s += steplen * ds;
    y(bound_idx) = s;
    w += steplen * dw;
    assert(PRETTY_SMALL > norm(y(free_idx)));

    if(verbose){
      cout <<"\tMean complementarity: " << mean_comp
           <<"\n\tStep length: " << steplen
           <<"\n\tCentering sigma: " << sigma << endl;
    }
  }
  if(verbose){
    cout << "Finished"
         <<"\n\t Final mean complementarity: " << mean_comp << endl;
  }
  return SolverResult(x,y,iter);
}
