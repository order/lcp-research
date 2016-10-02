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

  // 1D minimization:
  // <x+a*dx,y+a*dy> = <x,y> + a<x,dy> + a<y,dx> + a*a*<dx,dy> 
  //double alpha = -(dot(x,dy) + dot(y,dx)) / dot(dx,dy);
  //if(alpha <= 0)
  // alpha = 1.0; // 1D min doesn't make sense.
  
  return min(vec{1.0,
        //scale*alpha,
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
  uint N = q.n_elem;

  assert(N == x.n_elem);
  assert(N == y.n_elem);

  // Figure out what is free (-inf,inf)
  // and what is bound to be non-negative [0,inf)
  bvec free_vars = lcp.free_vars;
  uvec bound_idx = find(0 == free_vars);
  uvec free_idx = find(1 == free_vars);
  assert(N == bound_idx.n_elem + free_idx.n_elem);
  uint NB = bound_idx.n_elem; // number bound
  
  vec dx,dy,dir;
  // A matrix:
  // [Y  X]
  // [-M I]
  // Overwrite the Y,X diagonals every iteration
  // Zero out Y,X rows associated with free variables
  sp_mat A;
  vector<vector<sp_mat>> block_A;
  block_A.push_back(vector<sp_mat>{sp_mat(N,N),sp_mat(N,N)});
  block_A.push_back(vector<sp_mat>{-lcp.M - regularizer * speye(N,N),
        speye(N,N)});
  A = block_mat(block_A);

  vec b = vec(2*N);
  double mean_comp, steplen, sigma = 0.95;
  uint iter;
  for(iter = 0; iter < max_iter; iter++){
    if(verbose)
      cout << "---Iteration " << iter << "---" << endl;
    // Mean complementarity
    mean_comp = dot(x(bound_idx),y(bound_idx)) / (double) NB;
    if(mean_comp < comp_thresh)
      break;

    // Update A inplace
    // Only fill in bound indices
    for(uint i = 0; i < NB; i++){
      uint idx = bound_idx(i);
      A(idx,idx) = y(idx);
      A(idx,idx+N) = x(idx);
    }
                    
    // Form RHS from residual and complementarity
    b.head(N) = sigma * mean_comp - x % y;
    b(free_idx).fill(0); //Explicitly zero-out
    b.tail(N) = lcp.M * x + q - y;

    // Solve and extract directions
    dir = spsolve(A,
                  b,"superlu",opts);
    assert(2*N == dir.n_elem);
    dx = dir.head(N);
    dy = dir.tail(N);    

    steplen = steplen_heuristic(x(bound_idx),
                                y(bound_idx),
                                dx(bound_idx),
                                dy(bound_idx),
                                0.9);
    sigma = sigma_heuristic(sigma,steplen);

    x += steplen * dx;
    y += steplen * dy;

    if(verbose){
      cout <<"\t Mean complementarity: " << mean_comp
           <<"\n\t Step length: " << steplen
           <<"\n\t Centering sigma: " << sigma << endl;
    }
  }
  if(verbose){
    cout << "Finished"
         <<"\n\t Final mean complementarity: " << mean_comp << endl;
  }
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
  vec x = ones<vec>(N);
  vec y = ones<vec>(N);
  vec w;
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

  if(verbose)
    cout << "Forming pre-computed products..." << endl;
  mat PtP = mat(P.t() * P);
  assert(size(K,K) == size(PtP));
  mat PtPU = PtP*U;
  assert(size(K,N) == size(PtPU));
  mat PtPU_Pt = PtPU - P.t();
  assert(size(K,N) == size(PtPU_Pt));
  mat PtPUP = PtPU * P;
  assert(size(K,K) == size(PtPUP));

  vec Ptq = P.t() * q;
  if(verbose)
    cout << "Done..." << endl;
  
  assert(N == x.n_elem);
  assert(N == y.n_elem);
  assert(K == w.n_elem);

  vec dx,dy,dw,g,h,Pdw,S;
  mat A,G;
  
  double sparsity,mean_comp, steplen, sigma;
  sigma = 0.95;
  
  superlu_opts slu_opts;
  slu_opts.equilibrate = true; // This is important for conditioning
  slu_opts.permutation = superlu_opts::NATURAL;
  slu_opts.refine = superlu_opts::REF_NONE;

  uint iter;
  for(iter = 0; iter < max_iter; iter++){
    if(verbose)
      cout << "---Iteration " << iter << "---" << endl;
    // Mean complementarity
    mean_comp = dot(x,y) / (double) N;
    if(mean_comp < comp_thresh)
      break;

    // Generate reduced Netwon system
    S = x+y;
    g = sigma * mean_comp - x % y;
    assert(N == g.n_elem);
    
    A = PtPU_Pt * spdiag(1.0 / S);

     
    G = (A * spdiag(y)) * P - PtPUP;
    assert(size(K,K) == size(G));
    G += regularizer * speye(K,K);
    
    h = A*g + PtPU*x + Ptq - P.t()*y;
    assert(K == h.n_elem);

    // Solve (G,h) system
    // sparsity = (double) G.n_nonzero / (double) G.n_elem;
    // if(sparsity < 0.05){
    //   if(verbose)
    //     cout << "\tSolving (sparse)..." << endl;
    //   dw = spsolve(G,h,"superlu", slu_opts);
    //   // sparse solve
    // }
    // else{
    //   if(verbose)
    //     cout << "\tSolving (dense)..." << endl;
    //   dw = spsolve(G,h,"lapack");
    //   // dense solve
    // }
    

    // Options don't make much difference
    dw = arma::solve(G,h,solve_opts::equilibrate);
    assert(K == dw.n_elem);

    // Recover dy
    Pdw = plcp.P * dw;
    dy = (g - y % Pdw) / S;
    assert(N == dy.n_elem);
    
    // Recover dx
    dx = dy + Pdw;
    assert(N == dy.n_elem);    

    steplen = steplen_heuristic(x,y,dx,dy,0.9);
    sigma = sigma_heuristic(sigma,steplen);

    x += steplen * dx;
    y += steplen * dy;
    w += steplen * dw;

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
