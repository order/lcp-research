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
  double max_sigma = 0.995;
  double min_sigma = 0.1;
  if(steplen >= 0.8)
    sigma *= 0.975;
  else if(steplen < 0.2)
      sigma = 0.75 + 0.25*sigma;
  else if(steplen < 1e-3)
    sigma = max_sigma;
  return max(min_sigma,min(max_sigma,sigma));
}

////////////////////////////////////////////////////////
// Kojima

KojimaSolver::KojimaSolver(){
  comp_thresh = 1e-15;
  max_iter = 500;
  verbose = true;
}

// Really should be always using this.
pd_pair KojimaSolver::aug_solve(const LCP & lcp) const{
  uint N = lcp.q.n_elem;
  double scale = 1.0 * (double) N; // Balance feasibility and complementarity
  vec x,y;
  LCP alcp = augment_lcp(lcp,x,y,scale);
  solve(alcp,x,y);
  return make_pair(x,y);
}

void KojimaSolver::solve(const LCP & lcp,
                            vec & x,
                            vec & y) const{
  
  double regularizer = 1e-9;

  superlu_opts opts;
  opts.equilibrate = true;
  opts.permutation = superlu_opts::NATURAL;
  opts.refine = superlu_opts::REF_NONE;
  
  vec q = lcp.q;  
  uint N = q.n_elem;

  assert(N == x.n_elem);
  assert(N == y.n_elem);

  vec dx,dy,dir;

  sp_mat A;
  vector<vector<sp_mat>> block_A;
  block_A.push_back(vector<sp_mat>{sp_mat(N,N),sp_mat(N,N)});
  block_A.push_back(vector<sp_mat>{-lcp.M,speye(N,N)});
  A = block_mat(block_A) + regularizer*speye(2*N,2*N);

  vec b = vec(2*N);
  double mean_comp, steplen, sigma = 0.95;
  for(uint i = 0; i < max_iter; i++){
    // Mean complementarity
    mean_comp = dot(x,y) / (double) N;
    if(mean_comp < comp_thresh)
      break;

    // Update A inplace
    for(uint i = 0; i < N; i++){
      A(i,i) = y(i);
      A(i,i+N) = x (i);
    }
    
    // Form RHS from residual and complementarity
    b.head(N) = sigma * mean_comp - x % y;
    b.tail(N) = lcp.M * x + q - y;

    // Solve and extract directions
    dir = spsolve(A,b,"superlu",opts);
    assert(2*N == dir.n_elem);
    dx = dir.head(N);
    dy = dir.tail(N);    

    steplen = steplen_heuristic(x,y,dx,dy,0.9);
    sigma = sigma_heuristic(sigma,steplen);

    x += steplen * dx;
    y += steplen * dy;

    if(verbose){
      cout << "Iteration " << i
           <<"\n\t Mean complementarity: " << mean_comp
           <<"\n\t Step length: " << steplen
           <<"\n\t Centering sigma: " << sigma << endl;
    }
  }
  if(verbose){
    cout << "Finished"
         <<"\n\t Final mean complementarity: " << mean_comp << endl;
  }
}


/////////////////////////////////////////////////////////////
// PROJECTIVE

ProjectiveSolver::ProjectiveSolver(){
  comp_thresh = 1e-15;
  max_iter = 500;
  verbose = true;
}

// Really should be always using this.
pd_pair ProjectiveSolver::aug_solve(const PLCP & plcp) const{
  uint N = plcp.P.n_rows;
  double scale = 1.0 * (double) N; // Balance feasibility and complementarity
  vec x,y,w;
  PLCP aplcp = augment_plcp(plcp,x,y,w,scale);
  solve(aplcp,x,y,w);
  return make_pair(x,y);
}

void ProjectiveSolver::solve(const PLCP & plcp,
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
  
  double sparsity,mean_comp, steplen, sigma, regularizer;
  sigma = 0.99;
  regularizer = 1e-12;
  
  superlu_opts slu_opts;
  slu_opts.equilibrate = true; // This is important for conditioning
  slu_opts.permutation = superlu_opts::NATURAL;
  slu_opts.refine = superlu_opts::REF_NONE;
  
  for(uint i = 0; i < max_iter; i++){
    if(verbose)
      cout << "---Iteration " << i << "---" << endl;
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
}
