#include "hallway.h"
#include "lcp.h"

sp_mat build_hallway_P(const uint N,
                       const double p_stick,
                       const int action){
  assert(action == -1 or action == 1);
  sp_mat P =  p_stick * speye(N,N)
    + spdiag((1-p_stick) * ones<vec>(N-1),action);
  if(action < 0)
    P(0,N-1) = (1 - p_stick);
  else
    P(N-1,0) = (1 - p_stick);
  return P;
}

sp_mat build_hallway_M(const uint N,
                       const double p_stick,
                       const double gamma){
  sp_mat P_left = build_hallway_P(N,p_stick,-1);
  sp_mat P_right = build_hallway_P(N,p_stick,1);

  vector<sp_mat> blocks;
  blocks.push_back(speye(N,N) - gamma * P_left);
  blocks.push_back((1-gamma) * speye(N,N));
  blocks.push_back(speye(N,N) - gamma * P_right);

  return build_M(blocks);
}

vec build_hallway_q(const uint N){

  uint target = N / 2;
  vec costs = ones<vec>(N);
  costs(target) = 0;
  
  vec q = vec(4*N);
  q.head(N) = -ones<vec>(N) / (double) N;
  q.subvec(N,size(costs)) = costs;
  q.subvec(2*N,size(costs)) = costs;
  q.tail(N) = costs;

  return q;
}

LCP build_hallway_lcp(const uint N,
                      const double p_stick,
                      const double gamma){
  sp_mat M =  build_hallway_M(N,p_stick,gamma);
  vec q = build_hallway_q(N);

  assert(size(4*N,4*N) == size(M));
  assert(4*N == q.n_elem);

  return LCP(M,q);
}
