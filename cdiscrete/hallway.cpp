#include "hallway.h"
#include "lcp.h"

#include <assert.h>

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
  // Build the state-weights and costs
  vec points = linspace<vec>(0,1,N+1).head(N);
  //vec costs = ones<vec>(N);
  //costs(find(abs(points - 0.5) < 0.1)).fill(0);
  vec costs = min(ones<vec>(N), 16*pow(points - 0.5,2));
  mat q = mat(N,4);
  q.col(0) = -ones<vec>(N) / (double) N;
  q.col(1) = costs;
  q.col(2) = costs;
  q.col(3) = costs;

  return q;
}

sp_mat build_convolution_matrix(uint N, const vec & v){
  int n = v.n_elem;
  assert(1 == n %2); // Odd vectors only
  assert(n <= N);
  int n_2 = n / 2;

  umat loc = umat(2,n*N);
  vec data = vec(n*N);
  uint I = 0;
  for(int c = 0; c < N; c++){
    for(int i = 0; i < n; i++){
      int r = (N + c + i - n_2) % N;
      loc(0,I) = r;
      loc(1,I) = c;
      //cout << size(r,c) << endl;
      data(I++) = v(i);
    }
  }
  return sp_mat(loc,data,N,N);
}
