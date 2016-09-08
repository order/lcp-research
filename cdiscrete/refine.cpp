#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>
#include <armadillo>

#include "misc.h"
#include "io.h"
#include "mesh.h"
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
                 const vec & value){

  uint F = mesh.number_of_faces();
  
  mat bounds = mesh.find_box_boundary();
  vec lb = bounds.col(0);
  vec ub = bounds.col(1);
  
  mat grad = mesh.face_grad(value);
  assert(size(grad) == size(F,TRI_NUM_DIM));
  
  Points centers = mesh.get_face_centers();
  assert(size(centers) == size(F,TRI_NUM_DIM));
 
  Points post_0 = double_integrator(centers,-1,SIM_STEP);
  saturate(post_0,lb,ub);
  Points post_1 = double_integrator(centers,1,SIM_STEP);
  saturate(post_1,lb,ub);

  mat IP = mat(F,2);
  IP.col(0) = sum(grad % post_0,1);
  IP.col(1) = sum(grad % post_1,1);

  uvec policy = col_argmin(IP);
  assert(F == policy.n_elem);
  return policy;
}

uvec q_policy(const TriMesh & mesh,
              const vec & value,
              double gamma){

  uint F = mesh.number_of_faces();
  
  mat bounds = mesh.find_box_boundary();
  vec lb = bounds.col(0);
  vec ub = bounds.col(1);
  
  mat grad = mesh.face_grad(value);
  assert(size(grad) == size(F,TRI_NUM_DIM));
  
  Points centers = mesh.get_face_centers();
  assert(size(centers) == size(F,TRI_NUM_DIM));

  mat Q = build_di_costs(centers);
  
  Points post_0 = double_integrator(centers,-1,SIM_STEP);
  saturate(post_0,lb,ub);
  Points post_1 = double_integrator(centers,1,SIM_STEP);
  saturate(post_1,lb,ub);

  // Pad for the oob node
  vec padded_value = join_vert(value,vec({max(value)}));

  Q.col(0) += gamma * mesh.interpolate(post_0,padded_value);
  Q.col(1) += gamma * mesh.interpolate(post_1,padded_value);

  uvec policy = col_argmin(Q);
  assert(F == policy.n_elem);
  return policy;
}

int main(int argc, char** argv)
{
  po::variables_map var_map = read_command_line(argc,argv);

  string mesh_file = var_map["infile_base"].as<string>() + ".tri";
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

  // Find bounds
  mat bounds = mesh.find_box_boundary();
  vec lb = bounds.col(0);
  vec ub = bounds.col(1);
  cout << "\tLower bound:" << lb.t()
       << "\tUpper bound:" << ub.t();

  // Read in solution information
  string soln_file = var_map["infile_base"].as<string>() + ".sol";
  cout << "Reading in LCP solution file [" << soln_file << ']'  << endl;
  Unarchiver unarch(soln_file);
  vec p = unarch.load_vec("p");
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

  string out_file_base = var_map["outfile_base"].as<string>();

  // Heuristic calculations
  cout << "Calculating splitting heuristic..." << endl;
  cout << "\tValue and flow differences..." << endl;
  // Difference in flow function
  vec val_diff = mesh.face_diff(value);
  vec flow_diff = mesh.face_diff(flows.col(0)) + mesh.face_diff(flows.col(1));
  val_diff.save(out_file_base + ".val_diff.vec",raw_binary);
  flow_diff.save(out_file_base + ".flow_diff.vec",raw_binary);

  cout << "\tFace value gradient..."<<endl;
  mat grad = mesh.face_grad(value);
  vec grad_x = grad.col(0);
  vec grad_y = grad.col(1);
  grad_x.save(out_file_base + ".grad_x.vec",raw_binary);
  grad_y.save(out_file_base + ".grad_y.vec",raw_binary);

  cout << "\tFlow volume..." << endl;
  // Volume of the aggregate flow
  vec flow_vol = mesh.prism_volume(sum(flows,1));
  flow_vol.save(out_file_base +".flow_vol.vec",raw_binary);

  vec heuristic = val_diff % flow_vol;
  heuristic.save(out_file_base + ".heuristic.vec",raw_binary);

  assert(F == heuristic.n_elem);

  cout << "Finding policies" << endl;
  uvec f_pi = flow_policy(mesh,flows);  
  uvec g_pi = grad_policy(mesh,value);
  uvec q_pi = q_policy(mesh,value,0.997);

  uvec policy_agg = f_pi + 2*g_pi + 4*q_pi;
  policy_agg.save(out_file_base +".policy.uvec",raw_binary);
  
  cout << "Finding order statistic..." << endl;  
  vec sorted = arma::sort(heuristic);
  uint order = std::min(100,(int) std::ceil(0.05*F));
  double order_stat = sorted(F-order);

  TriMesh new_mesh(mesh);  

  cout << "Adding " << order << " new nodes..." << endl;
  Point center;

  policy_agg = vec_mod(policy_agg,7);
  for(uint f = 0; f < F; f++){
    if(heuristic(f) > order_stat or policy_agg(f) > 0){
      // Get center from old mesh
      center = convert(mesh.center_of_face(f));
      // Insert center into new mesh.
      new_mesh.insert(center);
    }
  }
  
  cout << "Refining..." << endl;
  new_mesh.refine(0.125,1.0);
  new_mesh.lloyd(25);
  new_mesh.freeze();

  cout << "Writing..."
       << "\n\tCGAL mesh file: " << out_file_base << ".tri"
       << "\n\tShewchuk node file: " << out_file_base << ".node"
       << "\n\tShewchuk ele file: " << out_file_base << ".ele" << endl;
  new_mesh.write_cgal(out_file_base + ".tri");
  new_mesh.write_shewchuk(out_file_base);
}
