#include "lcp.h"
#include "io.h"

LCP::LCP(const mat & aM,
         const vec & aq) : M(aM),q(aq){}

void LCP::write(const string & filename){
  Archiver arch;
  arch.add_sp_mat("M",M);
  arch.add_vec("q",q);
  arch.write(filename);
}

vector<sp_mat> build_E_blocks(const Simulator * sim,
                              const TriMesh & mesh,
                              double gamma,
                              bool strip_oob){
  Points points = mesh.get_spatial_nodes();
  uint N;
  if(strip oob)
    N = points.n_rows;
  else
    N = points.n_rows + 1;
  uint A = sim->num_actions();
  uint num_samples = 50;
  
  vector<sp_mat> E_blocks;
  sp_mat I = speye(N,N);
  mat actions = sim->get_actions();
  vec u;
  ElementDist P;
  for(uint i = 0; i < A; i++){
    P = ElementDist(N,N);
    u = actions.row(i).t();
    cout << "\tAction [" << i << "]: u = " << u.t() << endl;
    for(uint j = 0; j < num_samples; j++){
      P += hillcar.transition_matrix(mesh,u);
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
  for(vector<sp_mat>::const_iterator it = E_block.begin();
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

vec build_q(const Simulator * sim,
            const TriMesh & mesh,
            bool strip_oob){
  uint N = mesh.number_of_vertices();
  Points points = mesh.get_spatial_nodes();
  assert(size(N,2) == size(points));
  
  mat costs = sim->get_costs(points);
  uint A = sim->num_actions();
  assert(size(N,A) == size(costs));
 
  vec weights = sim->get_state_weights(points);
  assert(N == weights.n_elem);
  if(!strip_oob){
    double tail = 1.0 / (1.0 - gamma);
    costs = join_vert(costs,tail*ones<rowvec>(A));
    weights = join_vert(weights,zeros<vec>(1));
  }
  vec q = join_vert(-weights,vectorise(costs));
  return q;
}

LCP build_lcp(const Simulator * sim,
              const TriMesh & mesh,
              double gamma,
              bool strip_oob){
  cout << "Generating transition matrices..."<< endl;
  vector<sp_mat> E_blocks = build_E_blocks(sim,mesh,
                                           gamma,strip_oob);
  sp_mat M = build_M(E_blocks);
  vec q = build_q(sim,mesh,strip_oob);
  return LCP(M,q);
}
