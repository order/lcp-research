#include "di.h"
#include "misc.h"
#include "io.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

void generate_initial_mesh(const po::variables_map & var_map,
                           TriMesh & mesh){
  double B = var_map["boundary"].as<double>();    
  uint num_bang_points = var_map["bang_points"].as<uint>();
  double angle = var_map["mesh_angle"].as<double>();
  double length = var_map["mesh_length"].as<double>();
  
  vec lb = -B*ones<vec>(2);
  vec ub = B*ones<vec>(2);

  cout << "Initial meshing..."<< endl;  
  build_square_boundary(mesh,lb,ub);
  VertexHandle v_zero = mesh.insert(Point(0,0));

  if(num_bang_points > 0){
    add_di_bang_bang_curves(mesh,lb,ub,num_bang_points);
  }
  
  cout << "Refining based on (" << angle << "," << length <<  ") criterion ..."<< endl;
  mesh.refine(angle,length);
  
  cout << "Optimizing (10 rounds of Lloyd)..."<< endl;
  mesh.lloyd(10);
  mesh.refine(angle,length);

  mesh.freeze();

  // Write initial mesh to file
  string filename =  var_map["outfile_base"].as<string>();

  cout << "Writing:"
       << "\n\t" << (filename + ".node") << " (Shewchuk node file)"
       << "\n\t" << (filename + ".ele") << " (Shewchuk element file)"
       << "\n\t" << (filename + ".tri") << " (CGAL mesh file)" << endl;
  mesh.write_shewchuk(filename);
  mesh.write_cgal(filename + ".tri");
}

void build_lcp(const po::variables_map & var_map,
               const TriMesh & mesh){
  mat bounds = mesh.find_box_boundary();
  vec lb = bounds.col(0);
  vec ub = bounds.col(1);
  
  Points points = mesh.get_spatial_nodes();
  uint N = points.n_rows;
  
  cout << "Generating transition matrices..."<< endl;
  // Get transition matrices
  ElementDist P_pos = build_di_transition(points,
					  mesh,
					  lb,ub,1.0);
  ElementDist P_neg = build_di_transition(points,
					  mesh,
					  lb,ub,-1.0);

  double gamma = var_map["gamma"].as<double>();
  sp_mat E_pos = speye(N,N) + ( -gamma * P_pos);
  sp_mat E_neg = speye(N,N) - gamma * P_neg;
  
  sp_mat E_pos_t = E_pos.t();
  cout << "Building LCP..."<< endl;
  block_sp_row top = block_sp_row{sp_mat(),E_neg,E_pos};
  block_sp_row middle = block_sp_row{-E_neg.t(),sp_mat(),sp_mat()};
  block_sp_row bottom = block_sp_row{-E_pos.t(),sp_mat(),sp_mat()};
  block_sp_mat blk_M = block_sp_mat{top,middle,bottom};
  sp_mat M = bmat(blk_M);
  
  mat costs = build_di_costs(points);
  vec weights = build_di_state_weights(points);

  vec q = join_vert(-weights,vectorise(costs));
  assert(3*N == q.n_elem);

  string filename = var_map["outfile_base"].as<string>() + ".lcp";
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
    ("outfile_base,o", po::value<string>()->required(),
     "Prefix for all files generated")
    ("mesh_file,m", po::value<string>(), "Input (CGAL) mesh file")
    ("mesh_angle", po::value<double>()->default_value(0.125),
     "Mesh angle refinement criterion")
    ("mesh_length", po::value<double>()->default_value(1),
     "Mesh edge length refinement criterion")    
    ("gamma,g", po::value<double>()->default_value(0.997),
     "Discount factor")
    ("boundary,B", po::value<double>()->default_value(5.0),
     "Square boundary box [-B,B]^2")
    ("bang_points,b", po::value<uint>()->default_value(0),
     "Number of bang-bang curve points to add to initial mesh");
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

  TriMesh mesh;
  if(var_map.count("mesh_file")){
    string mesh_file = var_map["mesh_file"].as<string>();
    cout << "Reading in mesh from [" << mesh_file << "]..." << endl;
    mesh.read_cgal(mesh_file);
  }
  else{
    cout << "Generating initial mesh..." << endl;
    generate_initial_mesh(var_map,mesh);
  }
  mesh.freeze();

  mat bounds = mesh.find_box_boundary();
  vec lb = bounds.col(0);
  vec ub = bounds.col(1);
  
  cout << "Mesh stats:" << endl;
  cout << "\tNumber of vertices: " << mesh.number_of_vertices() << endl;
  cout << "\tNumber of faces: " << mesh.number_of_faces() << endl;
  cout << "\tLower bound:" << lb.t();
  cout << "\tUpper bound:" << ub.t();

  build_lcp(var_map,mesh);
}
