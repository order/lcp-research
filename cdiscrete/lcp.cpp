#include "lcp.h"
#include "io.h"
#include <assert.h>

LCP::LCP(const sp_mat & aM,
         const vec & aq) : M(aM),q(aq){}

void LCP::write(const string & filename){
  Archiver arch;
  arch.add_sp_mat("M",M);
  arch.add_vec("q",q);
  arch.write(filename);
}

vector<sp_mat> build_E_blocks(const Simulator * sim,
                              const Discretizer * disc,
                              double gamma,
                              bool include_oob){
  Points points = disc->get_spatial_nodes();
  uint N;
  if(include_oob)
    N = disc->number_of_all_nodes();
  else
    N = disc->number_of_spatial_nodes();
  
  uint A = sim->num_actions();
  uint Ad = sim->dim_actions();
  mat actions = sim->get_actions();
  assert(size(A,Ad) == size(actions));
  
  uint num_samples = 25;
  
  vector<sp_mat> E_blocks;
  sp_mat I = speye(N,N);  
  vec u;
  ElementDist P,Q;
  for(uint i = 0; i < A; i++){
    P = ElementDist(N,N);
    u = actions.row(i).t();
    cout << "\tAction [" << i << "]: u = " << u.t() << endl;
    for(uint j = 0; j < num_samples; j++){
      P += sim->transition_matrix(disc,u,include_oob);
    }
    P /= (double) num_samples;
    E_blocks.push_back(I - gamma * P);
  }
  assert(A == E_blocks.size());
  return E_blocks;
}

sp_mat build_M(const vector<sp_mat> & E_blocks){
  block_sp_mat blk_M;
  block_sp_row tmp_row;
  blk_M.push_back(block_sp_row{sp_mat()});
  uint A = E_blocks.size();
  for(vector<sp_mat>::const_iterator it = E_blocks.begin();
      it != E_blocks.end(); ++it){
    // Build -E.T row
    tmp_row.clear();
    tmp_row.push_back(-it->t());
    for(uint j = 0; j < A; j++){
      tmp_row.push_back(sp_mat());
    }
    assert(A + 1 == tmp_row.size());
    // Add to end of growing matrix
    blk_M.push_back(tmp_row);

    // Add E to top row
    blk_M.at(0).push_back(*it);
  }
  assert(A+1 == blk_M.at(0).size());
  assert(A+1 == blk_M.size());
  
  sp_mat M = bmat(blk_M);
  assert(M.is_square());
  return M;
}

vec build_q_vec(const Simulator * sim,
                const Discretizer * disc,
                double gamma,
                bool include_oob){
  uint N = disc->number_of_all_nodes();
  uint n = disc->number_of_spatial_nodes(); // One oob nodes
  
  Points points = disc->get_spatial_nodes();
  
  mat costs = sim->get_costs(points);
  uint A = sim->num_actions();
  assert(size(n,A) == size(costs));

  if(include_oob){
    double tail = 1.0 / (1.0 - gamma);
    costs = join_vert(costs,tail*ones<rowvec>(A));
    assert(size(N,A) == size(costs));
  }

 
  vec weights = sim->get_state_weights(points);
  assert(n == weights.n_elem);
  if(include_oob){
    weights = join_vert(weights,zeros<vec>(1));
    assert(N == weights.n_elem);
  }
  
  vec q = join_vert(-weights,vectorise(costs));
  if(include_oob)
    assert((A+1)*N == q.n_elem);
  else
    assert((A+1)*n == q.n_elem);
  return q;
}

LCP build_lcp(const Simulator * sim,
              const Discretizer * disc,
              double gamma,
              bool include_oob){
  cout << "Generating transition matrices..."<< endl;
  vector<sp_mat> E_blocks = build_E_blocks(sim,disc,
                                           gamma,include_oob);
  sp_mat M = build_M(E_blocks);
  vec q = build_q_vec(sim,disc,gamma,include_oob);
  assert(q.n_elem == M.n_rows);
  assert(q.n_elem == M.n_cols);

  return LCP(M,q);
}
