#include "hillcar.h"
#include "misc.h"
#include "io.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

mat hillcar_bounding_box(){
  return {{-6,2},{-5,5}};
}

HillcarSimulator build_hillcar_simulator(const po::variables_map & var_map){
  mat bbox = hillcar_bounding_box();
  mat actions = vec{-2,2};
  return HillcarSimulator(bbox,actions);
}

void generate_initial_mesh(const po::variables_map & var_map,
                           TriMesh & mesh){
  double angle = var_map["mesh_angle"].as<double>();
  double length = var_map["mesh_length"].as<double>();

  cout << "Initial meshing..."<< endl;
  mat bbox = hillcar_bounding_box();
  mesh.build_box_boundary(bbox);
  VertexHandle v_zero = mesh.insert(Point(0,0));
  
  cout << "Refining based on (" << angle
       << "," << length <<  ") criterion ..."<< endl;
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

// This should be generalized, I think
void build_lcp(const po::variables_map & var_map,
               const HillcarSimulator & hillcar,
               const TriMesh & mesh){
  vec lb,ub;
  mat bounds = hillcar.get_bounding_box();
  
  Points points = mesh.get_spatial_nodes();
  uint N = points.n_rows;

  uint NUM_ACTIONS = 2; // [-1,1]
  uint NUM_SAMPLES = 150; // Read from var_map?
  
  vector<sp_mat> E_vector;
  vec u;
  sp_mat I = speye(N+1,N+1); // Spatial + OOB
  double gamma = var_map["gamma"].as<double>();
  cout << "Generating transition matrices..."<< endl;
  mat actions = hillcar.get_actions();
  for(uint i = 0; i < NUM_ACTIONS; i++){
    ElementDist P = ElementDist(N+1,N+1);
    u = actions.row(i).t();
    cout << "\tAction [" << i << "]: u = " << u.t() << endl;
    for(uint j = 0; j < NUM_SAMPLES; j++){
      P += hillcar.transition_matrix(mesh,u);
    }
    P /= (double)NUM_SAMPLES;    
    E_vector.push_back(I - gamma * P);
  }  

  cout << "Building LCP..."<< endl;
  block_sp_row top = block_sp_row{sp_mat(),E_vector[0],E_vector[1]};
  block_sp_row middle = block_sp_row{-E_vector[0].t(),sp_mat(),sp_mat()};
  block_sp_row bottom = block_sp_row{-E_vector[1].t(),sp_mat(),sp_mat()};
  block_sp_mat blk_M = block_sp_mat{top,middle,bottom};
  sp_mat M = bmat(blk_M);
  
  mat costs = hillcar.get_costs(points);
  assert(N == costs.n_rows);
  // Append OOB cost
  double tail = 1.0 / (1.0 - gamma);
  costs = join_vert(costs,tail*ones<rowvec>(NUM_ACTIONS));

  vec center = {-4.0,0.0};
  vec weights = lp_norm_weights(points - repmat(center.t(),N,1),2);
  // Append OOB cost
  weights = join_vert(weights,zeros<vec>(1));

  vec q = join_vert(-weights,vectorise(costs));
  assert(3*(N+1) == q.n_elem);

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

  HillcarSimulator hillcar = build_hillcar_simulator(var_map);
  
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

  mat bbox = hillcar.get_bounding_box();
  
  cout << "Mesh stats:" << endl;
  cout << "\tNumber of vertices: " << mesh.number_of_vertices() << endl;
  cout << "\tNumber of faces: " << mesh.number_of_faces() << endl;
  cout << "\tLower bound:" << bbox.col(0).t();
  cout << "\tUpper bound:" << bbox.col(1).t();

  build_lcp(var_map,hillcar,mesh);
}
