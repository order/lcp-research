#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>
#include <armadillo>

#include "misc.h"
#include "io.h"
#include "tet_mesh.h"
#include "dubins.h"
#include "refine.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;
using namespace arma;
using namespace tet_mesh;
using namespace dubins;

po::variables_map read_command_line(uint argc, char** argv){
  po::options_description desc("Meshing options");
  desc.add_options()
    ("help", "produce help message")
    ("infile_base,i", po::value<string>()->required(), "Input file base")
    ("outfile_base,o", po::value<string>()->required(),
     "Output (CGAL) mesh file base")
    ("mesh_angle", po::value<double>()->default_value(0.125),
     "Mesh angle refinement criterion")
    ("mesh_length", po::value<double>()->default_value(0.5),
     "Mesh edge length refinement criterion")    
    ("max_expansion", po::value<int>()->default_value(100),
     "Max number of cells to split");
  po::variables_map var_map;
  po::store(po::parse_command_line(argc, argv, desc), var_map);
  po::notify(var_map);

  if (var_map.count("help")) {
    cout << desc << "\n";
    exit(1);
  }
  return var_map;
}

////////////////////////////////////////////////////////////
// MAIN FUNCTION ///////////////////////////////////////////
////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  po::variables_map var_map = read_command_line(argc,argv);

  string mesh_file = var_map["infile_base"].as<string>() + ".ctri";

  // Read in the CGAL mesh
  TetMesh mesh;
  cout << "Reading in cgal mesh file [" << mesh_file << ']'  << endl;
  mesh.read_cgal(mesh_file);
  mesh.freeze();
  uint V = mesh.number_of_spatial_nodes();
  uint N = mesh.number_of_all_nodes();
  assert(N == V+1);
  uint C = mesh.number_of_cells();
  cout << "Mesh stats:"
       << "\n\tNumber of vertices: " << V
       << "\n\tNumber of tetrahedra: " << C
       << endl;

  // Find boundary from the mesh and create the simulator object
  DubinsCarSimulator dubins = DubinsCarSimulator(DUBINS_ACTIONS);

  // Read in solution information
  string soln_file = var_map["infile_base"].as<string>() + ".sol";
  cout << "Reading in LCP solution file [" << soln_file << ']'  << endl;
  Unarchiver unarch(soln_file);
  vec p = unarch.load_vec("p");

  Archiver arch;
  
  // Make sure that the primal information makes sense
  mat P;
  assert(0 == p.n_elem % N);
  uint A = p.n_elem / N;
  cout << "Blocking primal solution..."
       << "\n\tLength of primal solution: " << p.n_elem
       << "\n\tRatio of primal length to vertex number: " << A << endl;
  assert((DUBINS_ACTIONS.n_rows+1) == A);

  P = reshape(p,size(N,A));
  P = P.head_rows(V); // Prune off OOB
  vec value = P.col(0);
  mat flows = P.tail_cols(2);

  // Heuristic calculations
  cout << "Calculating splitting heuristic..." << endl;
  // Policy disagreement
  cout << "Finding policy disagreements..." << endl;
  uvec f_pi = flow_policy(&mesh,flows);  
  uvec g_pi = grad_policy(&mesh,&dubins,value);
  uvec q_pi = q_policy(&mesh,&dubins,value,DUBINS_GAMMA);
  uvec policy_agg = f_pi + g_pi  + q_pi;
  arch.add_uvec("policy_agg",policy_agg);
  policy_agg = vec_mod(policy_agg,3); // 0 if all policies agree

  mat grad = mesh.cell_gradient(value);
  arch.add_vec("grad_x",grad.col(0));
  arch.add_vec("grad_y",grad.col(1));

  // Bellman residual
  cout << "\tBellman residual at centroids..." << endl;
  vec bell_res = bellman_residual(&mesh,&dubins,value,DUBINS_GAMMA,0,25);
  arch.add_vec("bellman_residual",bell_res);

  // Advantage function
  //vec adv_res = advantage_residual(&mesh,&dubins,value,DUBINS_GAMMA,15);
  //arch.add_vec("advantage_residual",adv_res);
  
  vec adv_fun = advantage_function(&mesh,&dubins,value,DUBINS_GAMMA,0,25);
  arch.add_vec("advantage_function",adv_fun);

  vec vol = mesh.cell_volume();
  arch.add_vec("volume",vol);
  
  // Volume of the aggregate flow
  cout << "\tFlow volume..." << endl;
  vec flow_vol = mesh.prism_volume(sum(flows,1));
  arch.add_vec("flow_vol",flow_vol);

  vec heuristic_1 = bell_res % sqrt(flow_vol);
  arch.add_vec("heuristic_1",heuristic_1);

  double q1 = 1.0 - 50.0 / (double)V;
  q1 = min(1.0,max(0.9,q1));
  cout << "Finding heuristic_1 " << q1 << " quantile..." << endl;
  double cutoff_1 = quantile(heuristic_1, q1);
  cout << "\t Cutoff: " << cutoff_1
       << "\n\t Max:" << max(heuristic_1)
       << "\n\t Min:" << min(heuristic_1) << endl;

  vec heuristic_2 = adv_fun % flow_vol;
  heuristic_2(find(policy_agg == 0)).fill(0);
  assert(C == heuristic_2.n_elem);
  arch.add_vec("heuristic_2",heuristic_2); 

  double q2 = 1.0 - 100.0 / (double)V;
  q2 = min(1.0,max(0.9,q1));
  cout << "Finding heuristic_2 " << q2 << " quantile..." << endl;
  double cutoff_2 = quantile(heuristic_2,q2);
  cout << "\t Cutoff_2: " << cutoff_2
       << "\n\t Max:" << max(heuristic_2)
       << "\n\t Min:" << min(heuristic_2) << endl;
  cutoff_2 = max(cutoff_2,1e-5);

  // Split the cells if they have a large heuristic_1 or
  // policies disagree on them.
  TetMesh new_mesh(mesh);  
  Point center;
  uint new_nodes = 0;
  for(uint c = 0; c < C; c++){
    if(vol(c) < 0.005){
      continue;
    }
    
    if(heuristic_1(c) > cutoff_1
       or heuristic_2(c) > cutoff_2){
      //or policy_agg(f) != 0){
      // Get center from old mesh
      center = vec_to_point(mesh.center_of_cell(c));
      // Insert center into new mesh.
      new_mesh.insert(center);
      new_nodes++;
    }
  }
  cout << "Added " << new_nodes << " new nodes..." << endl;
  new_mesh.freeze();

  // Write out all the information
  string out_file_base = var_map["outfile_base"].as<string>();
  arch.write(out_file_base + ".stats");
  cout << "Writing..."
       << "\n\tCGAL mesh file: " << out_file_base << ".ctri" << endl;
  new_mesh.write_cgal(out_file_base + ".ctri");
}
