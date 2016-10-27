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

vector<sp_mat> make_freebie_flow_bases(const sp_mat & value_basis,
                               const vector<sp_mat> blocks){
  // TODO: think more carefully about q;
  // Ignore? Project onto value basis? Add to value basis for flow calc?
  vector<sp_mat> flow_bases;
  uint A = blocks.size();
  for(uint a = 0; a < A; a++){
    sp_mat raw_basis = blocks.at(a) * value_basis;
    flow_bases.push_back(sp_mat(orth(mat(raw_basis))));
    // Orthonorm (TODO: do directly in sparse?)
  }
  return flow_bases;
}


// Should be pretty generic; put in a more general file
PLCP approx_lcp(const vec & points,
                const sp_mat & value_basis,
                        const block_sp_vec & blocks,
                        const mat & Q,
                        const bvec & free_vars){
  uint n = points.n_elem;
  uint A = blocks.size();
  uint N = n*(A+1);
  
  assert(A > 0);
  assert(size(n,n) == size(blocks.at(0)));
  assert(size(n,A+1) == size(Q));
    
  vector<sp_mat> flow_bases = make_freebie_flow_bases(value_basis,
                                                      blocks);
  assert(A == flow_bases.size());

  // Assemble basis blocks into full basis
  block_sp_vec p_blocks;
  p_blocks.reserve(A + 1);
  p_blocks.push_back(value_basis);
  p_blocks.insert(p_blocks.end(),
                  flow_bases.begin(),
                  flow_bases.end());
  assert((A+1) == p_blocks.size());
  sp_mat P = block_diag(p_blocks);

  // Build the U vector based on P
  // TODO: could make more efficient by working with block
  // and not assembling into intermediate M
  vec q = vectorise(Q);
  assert(N == q.n_elem);
  
  sp_mat M = build_M(blocks) + 1e-10 * speye(N,N); // Regularize
  sp_mat U = P.t() * M * P * P.t();

  //bvec free_vars = zeros<bvec>(V); // TODO
  PLCP plcp = PLCP(P,U,q,free_vars);
  return plcp;
}
