#include "hallway.h"
#include "lcp.h"

sp_mat build_hallway_P(const uint N,
                       const double p_stick,
                       const int action){
  // Build the hallway transition matrix associated with
  // and action
  
  assert(action == -1 or action == 1);
  sp_mat P =  p_stick * speye(N,N)
    + spdiag((1-p_stick) * ones<vec>(N-1),action);
  if(action < 0)
    P(0,N-1) = (1 - p_stick);
  else
    P(N-1,0) = (1 - p_stick);
  return P;
}

vector<sp_mat> build_hallway_blocks(const uint N,
                       const double p_stick,
                       const double gamma){
  // Build the transition matrices for -1,0,1
  vector<sp_mat> blocks;
  // Left
  sp_mat P_left = build_hallway_P(N,p_stick,-1);
  blocks.push_back(speye(N,N) - gamma * P_left);
  
  // Stay
  blocks.push_back((1-gamma) * speye(N,N));

  //Right
  sp_mat P_right = build_hallway_P(N,p_stick,1);
  blocks.push_back(speye(N,N) - gamma * P_right);
  return blocks;
}

mat build_hallway_q(const uint N){
  arma_rng::set_seed(1);

  // Build the state-weights and costs
  vec points = linspace<vec>(0,1,N+1).head(N);
  vec costs = ones<vec>(N);
  costs(find(abs(points - 0.5) < 0.1)).fill(0);
  
  mat q = mat(N,4);
  double std = 0.2;
  q.col(0) = -ones<vec>(N) / (double) N;
  q.col(1) = costs + std * (2 * randu<vec>(N) - 1);
  q.col(2) = costs + std * (2 * randu<vec>(N) - 1);
  q.col(3) = costs + std * (2 * randu<vec>(N) - 1);

  return q;
}

LCP build_hallway_lcp(const uint N,
                      const double p_stick,
                      const double gamma){
  // Build the exact LCP
  sp_mat M =  build_M(build_hallway_blocks(N,p_stick,gamma));
  mat q = build_hallway_q(N);
  assert(size(4*N,4*N) == size(M));
  assert(size(N,4) == size(q));
  bvec free_var = zeros<bvec>(4*N);
  free_var.head(N).fill(1);
  return LCP(M,vectorise(q),free_var);
}

sp_mat build_smoothed_identity(uint N,double p){
  // A tridiagonal matrix that smoothes out constraints
  assert(p > 0);
  assert(p <= 1);
  double q = (1.0 - p) / 2.0;

  sp_mat I = p*speye(N,N)
    + q*spdiag(ones<vec>(N-1),1)
    + q*spdiag(ones<vec>(N-1),-1);
  I(0,N-1) = q;
  I(N-1,0) = q;
  return I;
}

LCP build_smoothed_hallway_lcp(const uint N,
                               const double p_stick,
                               const double p_smooth,
                               const double gamma){

  vector<sp_mat> blocks = build_hallway_blocks(N,p_stick,gamma);
  uint A = blocks.size();
  assert(3 == A);
  
  mat q = build_hallway_q(N);
  assert(size(N,A+1) == size(q));

  sp_mat smoother = build_smoothed_identity(N,p_smooth);
  for(uint a = 0; a< A; a++){
    blocks[a] = smoother * blocks[a];
    q.col(a+1) = smoother * q.col(a+1);
  }
  
  sp_mat M =  build_M(blocks);
  bvec free_var = zeros<bvec>((A+1)*N);
  free_var.head(N).fill(1);
  return LCP(M,vectorise(q),free_var);
}
