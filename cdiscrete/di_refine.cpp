#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>
#include <armadillo>

#include "misc.h"
#include "io.h"
#include "tri_mesh.h"
#include "di.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;
using namespace arma;


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
     "Max number of cells to split")
    ("perc_expansion",po::value<double>()->default_value(0.1),
     "Percentage of cells to expand");
  po::variables_map var_map;
  po::store(po::parse_command_line(argc, argv, desc), var_map);
  po::notify(var_map);

  if (var_map.count("help")) {
    cout << desc << "\n";
    exit(1);
  }
  return var_map;
}

vec bellman_residual(const TriMesh & mesh,
                     const DoubleIntegratorSimulator & di,
                     const vec & values,
                     double gamma,
                     uint samples = 25){
    
  // Pad for the oob node
  double max_val = 1.0 / (1.0 - gamma);
  vec padded_values = join_vert(values, vec{max_val});
   
  Points centers = mesh.get_face_centers();
  vec v_interp = mesh.interpolate(centers,padded_values);
  mat Q = estimate_Q(centers,
                     &mesh,
                     &di,
                     padded_values,
                     gamma,
                     samples);
                       
  // TODO: think about weighting by flows at center
  // Combines flow and values
  vec v_q_est = min(Q,1);
  uint F = mesh.number_of_faces();
  assert(F == v_q_est.n_elem);
  
  return abs(v_interp - v_q_est);
}

// Simplest policy; return action with max flow
uvec flow_policy(const TriMesh & mesh,
                 const mat & flows){
  Points centers = mesh.get_face_centers();
  // Pad with 0 flow at oob node
  mat padded_flows = join_vert(flows,zeros<rowvec>(2));
 
  mat interp = mesh.interpolate(centers,padded_flows);
  uvec policy = col_argmax(interp);
  assert(centers.n_rows == policy.n_elem);
  return policy;
}

uvec grad_policy(const TriMesh & mesh,
                 const DoubleIntegratorSimulator & di,
                 const vec & value,
                 uint samples=25){

  uint F = mesh.number_of_faces();
  
  mat bounds = mesh.find_bounding_box();
  vec lb = bounds.col(0);
  vec ub = bounds.col(1);
  
  mat grad = mesh.face_grad(value);
  assert(size(grad) == size(F,TRI_NUM_DIM));
  
  Points centers = mesh.get_face_centers();
  assert(size(centers) == size(F,TRI_NUM_DIM));

  assert(2 == di.num_actions());
  assert(1 == di.dim_actions());
  uint A = di.num_actions();
  mat actions = di.get_actions();
  
  mat IP = zeros<mat>(F,A);
  for(uint a = 0; a < A; a++){
    vec action = actions.row(a).t();
    for(uint s = 0; s < samples; s++){
      Points p_next = di.next(centers,action);
      IP.col(a) += sum(grad % p_next,1);
    }
  }

  uvec policy = col_argmin(IP);
  assert(F == policy.n_elem);
  return policy;
}

uvec q_policy(const TriMesh & mesh,
              const DoubleIntegratorSimulator& di,
              const vec & values,
              double gamma,
              uint samples=25){

  uint F = mesh.number_of_faces();
  double max_val = 1.0 / (1.0 - gamma);
  vec padded_values = join_vert(values,
                                vec{max_val});
      
  Points centers = mesh.get_face_centers();
  assert(size(centers) == size(F,TRI_NUM_DIM));
  

  

  mat Q = estimate_Q(centers,
                     &mesh,
                     &di,
                     padded_values,
                     gamma,
                     samples);

  uvec policy = col_argmin(Q);
  assert(F == policy.n_elem);
  return policy;
}

////////////////////////////////////////////////////////////
// MAIN FUNCTION ///////////////////////////////////////////
////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  po::variables_map var_map = read_command_line(argc,argv);

  string mesh_file = var_map["infile_base"].as<string>() + ".tri";

  // Read in the CGAL mesh
  TriMesh mesh;
  cout << "Reading in cgal mesh file [" << mesh_file << ']'  << endl;
  mesh.read_cgal(mesh_file);
  mesh.freeze();
  uint V = mesh.number_of_vertices();
  uint F = mesh.number_of_faces();
  cout << "Mesh stats:"
       << "\n\tNumber of vertices: " << V
       << "\n\tNumber of faces: " << F
       << endl;

  // Find boundary from the mesh and create the simulator object
  mat bbox = mesh.find_bounding_box();
  vec lb = bbox.col(0);
  vec ub = bbox.col(1);
  cout << "\tLower bound:" << lb.t()
       << "\tUpper bound:" << ub.t();
  DoubleIntegratorSimulator di = DoubleIntegratorSimulator(bbox,vec{-1,1});

  // Read in solution information
  string soln_file = var_map["infile_base"].as<string>() + ".sol";
  cout << "Reading in LCP solution file [" << soln_file << ']'  << endl;
  Unarchiver unarch(soln_file);
  vec p = unarch.load_vec("p");

  // Make sure that the primal information makes sense
  double p_over_v = ((double)p.n_elem / (double) V);
  uint rem = p.n_elem % V;
  cout << "Blocking primal solution..."
       << "\n\tLength of primal solution: " << p.n_elem
       << "\n\tRatio of primal length to vertex number: " << p_over_v
       << "\n\tRemainder of above: " << rem << endl;
  assert(p.n_elem == 3*V);  
  mat P = reshape(p,size(V,3));
  vec value = P.col(0);
  mat flows = P.tail_cols(2);

  
  // Heuristic calculations
  Archiver arch;
  cout << "Calculating splitting heuristic..." << endl;
  cout << "\tBellman residual at centroids..." << endl;
  vec residual = bellman_residual(mesh,di,value,0.997);
  arch.add_vec("residual",residual);

  cout << "\tFlow volume..." << endl;
  // Volume of the aggregate flow
  vec flow_vol = mesh.prism_volume(sum(flows,1));
  arch.add_vec("flow_vol",flow_vol);
  
  vec heuristic = residual % flow_vol;
  assert(F == heuristic.n_elem);
  arch.add_vec("heuristic",heuristic);

  // Policy disagreement
  cout << "Finding policies" << endl;
  uvec f_pi = flow_policy(mesh,flows);  
  uvec g_pi = grad_policy(mesh,di,value);
  uvec q_pi = q_policy(mesh,di,value,0.997);
  uvec policy_agg = f_pi + 2*g_pi + 4*q_pi;
  arch.add_uvec("policy_agg",policy_agg);
  
  cout << "Finding heuristic 0.98 quantile..." << endl;
  double cutoff = quantile(heuristic,0.98);  
  cout << "\t Cutoff: " << cutoff
       << "\n\t Max:" << max(heuristic)
       << "\n\t Min:" << min(heuristic) << endl;

  // Split the cells if they have a large heuristic or
  // policies disagree on them.
  TriMesh new_mesh(mesh);  
  Point center;
  uint new_nodes = 0;
  policy_agg = vec_mod(policy_agg,7); // All policies agree
  for(uint f = 0; f < F; f++){
    if(heuristic(f) > cutoff or policy_agg(f) > 0){
      // Get center from old mesh
      center = convert(mesh.center_of_face(f));
      // Insert center into new mesh.
      new_mesh.insert(center);
      new_nodes++;
    }
  }
  cout << "Added " << new_nodes << " new nodes..." << endl;
  
  cout << "Refining..." << endl;
  new_mesh.refine(0.125,1.0);
  new_mesh.lloyd(25);
  new_mesh.freeze();

  // Write out all the information
  string out_file_base = var_map["outfile_base"].as<string>();
  arch.write(out_file_base + ".stats");
  cout << "Writing..."
       << "\n\tCGAL mesh file: " << out_file_base << ".tri"
       << "\n\tShewchuk node file: " << out_file_base << ".node"
       << "\n\tShewchuk ele file: " << out_file_base << ".ele" << endl;
  new_mesh.write_cgal(out_file_base + ".tri");
  new_mesh.write_shewchuk(out_file_base);
}
