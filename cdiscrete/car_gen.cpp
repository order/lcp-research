#include "car.h"
#include "misc.h"
#include "io.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

void build_lcp(const po::variables_map & var_map,
               const TetMesh & mesh){
  
  Points points = mesh.get_spatial_nodes();
  uint N = points.n_rows;
  
  double gamma = var_map["gamma"].as<double>();
  cout << "Generating transition matrices..."
       <<"\n\tUsing gamma = " << gamma << endl;
  // Get transition matrices
  double u1,u2;
  vector<sp_mat> E_vector;
  ElementDist P;
  sp_mat I = speye(N+1,N+1);
  double oob_self_prob = 0;
  uint a_id = 0;
  int start,finish;
  for(int i = 0; i < 2; ++i){
    u1 = i;
    if(i == 0){
      // Can't turn without forward velocity
      start = 0; finish = 1;
    }
    else{
      start = -1; finish = 2;
    }
    for(int j = start; j < finish; ++j){
      u2 = j;
      cout << "\t Building matrix " << a_id
           << ": action = (" << u1 << "," << u2 << ")\n";
      ElementDist P = build_car_transition(points,mesh,
                                           u1,u2,oob_self_prob);
      assert(size(N+1,N+1) == size(P));
      E_vector.push_back(I - gamma*P);
      a_id++;
    }
  }
  
  assert(4 == a_id);
  uint NUM_ACTIONS = a_id;
  
  cout << "Building LCP..."<< endl;
  block_sp_mat blk_M;
  block_sp_row tmp_row;
  blk_M.push_back(block_sp_row{sp_mat()}); 
  for(uint i = 0; i < NUM_ACTIONS; i++){
    // Build -E.T row
    tmp_row.clear();
    tmp_row.push_back(-E_vector.at(i).t());
    for(uint j = 0; j < NUM_ACTIONS; j++){
      tmp_row.push_back(sp_mat());
    }
    assert(NUM_ACTIONS + 1 == tmp_row.size());
    // Add to end of growing matrix
    blk_M.push_back(tmp_row);

    // Add E to top row
    blk_M.at(0).push_back(E_vector.at(i));
  }
  assert(NUM_ACTIONS+1 == blk_M.at(0).size());
  assert(NUM_ACTIONS+1 == blk_M.size());
  sp_mat M = bmat(blk_M);
  assert(M.is_square());
  assert((NUM_ACTIONS+1)*(N+1) == M.n_rows);

  // Get costs for spatial nodes, and add oob node
  double tail_factor = 1.0 / (1.0 - gamma);
  mat costs = build_car_costs(points,
                              NUM_ACTIONS, tail_factor);
  costs = join_vert(costs,tail_factor * ones<mat>(1,NUM_ACTIONS));

  // Get weights for spatial nodes, and add oob node
  vec weights = build_car_state_weights(points);
  weights = join_vert(weights,zeros<vec>(1));
  
  vec q = join_vert(-weights,vectorise(costs));
  assert(q.n_elem == M.n_rows);

  string filename = var_map["lcp"].as<string>();
  cout << "Writing " << filename << endl;
  Archiver archiver;
  archiver.add_sp_mat("M",M);
  archiver.add_vec("q",q);
  archiver.write(filename);
}

po::variables_map read_command_line(uint argc, char** argv){
  po::options_description desc("Meshing options");
  desc.add_options()
    ("help", "produce help message")
    ("lcp,l", po::value<string>()->required(),
     "LCP out file")
    ("mesh,m", po::value<string>()->required(),
     "Input (CGAL) mesh file")
    ("gamma,g", po::value<double>()->default_value(0.997),
     "Discount factor");
  po::variables_map var_map;
  po::store(po::parse_command_line(argc, argv, desc), var_map);
  po::notify(var_map);

  if (var_map.count("help")) {
    cout << desc << "\n";
    exit(1);
  }
  return var_map;
}

//===========================================================
// Main function

int main(int argc, char** argv)
{
  po::variables_map var_map = read_command_line(argc,argv);

  TetMesh mesh;
  string mesh_file = var_map["mesh"].as<string>();
  cout << "Reading in mesh from [" << mesh_file << "]..." << endl;
  mesh.read_cgal(mesh_file);
  mesh.freeze();
  
  cout << "Mesh stats:" << endl;
  cout << "\tNumber of vertices: " << mesh.number_of_vertices() << endl;
  cout << "\tNumber of tetrahedra: " << mesh.number_of_cells() << endl;
  mat bounds = mesh.find_box_boundary();
  vec lb = bounds.col(0);
  vec ub = bounds.col(1);
  cout << "\tLower bound:" << lb.t();
  cout << "\tUpper bound:" << ub.t();

  build_lcp(var_map,mesh);
}
