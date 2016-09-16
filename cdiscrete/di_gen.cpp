#include "di.h"
#include "misc.h"
#include "io.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

DoubleIntegratorSimulator build_di_simulator(const po::variables_map & var_map){
  double B = var_map["boundary"].as<double>();
  mat bbox = mat(2,2);
  bbox.col(0).fill(-B);
  bbox.col(1).fill(B);
  mat actions = vec{-1,1};
  return DoubleIntegratorSimulator(bbox,actions);
}

void generate_initial_mesh(const po::variables_map & var_map,
                           const DoubleIntegratorSimulator & di,
                           TriMesh & mesh){
  uint num_bang_points = var_map["bang_points"].as<uint>();
  double angle = var_map["mesh_angle"].as<double>();
  double length = var_map["mesh_length"].as<double>();

  double B = var_map["boundary"].as<double>();    
  vec lb = -B*ones<vec>(2);
  vec ub = B*ones<vec>(2);
  mat bbox = join_horiz(lb,ub);

  cout << "Initial meshing..."<< endl;
  mesh.build_box_boundary(lb,ub);
  VertexHandle v_zero = mesh.insert(Point(0,0));

  if(num_bang_points > 0){
    di.add_bang_bang_curve(mesh,num_bang_points);
  }
  
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

void build_lcp(const po::variables_map & var_map,
               const DoubleIntegratorSimulator & di,
               const TriMesh & mesh){
  vec lb,ub;
  mat bounds = di.get_bounding_box();
  
  Points points = mesh.get_spatial_nodes();
  uint N = points.n_rows;

  uint NUM_ACTIONS = 2; // [-1,1]
  uint NUM_SAMPLES = 150; // Read from var_map?
  
  vector<sp_mat> E_vector;
  double u;
  sp_mat I = speye(N,N);
  double gamma = var_map["gamma"].as<double>();
  std::default_random_engine rand_gen;
  std::normal_distribution<double> randn(0.0,0.1);
  cout << "Generating transition matrices..."<< endl;
  for(uint i = 0; i < NUM_ACTIONS; i++){
    ElementDist P = ElementDist(N,N);
    u = 2.0 * i - 1.0;
    cout << "\tAction [" << i << "]: u = " << u << endl;
    for(uint j = 0; j < NUM_SAMPLES; j++){
      P += di.transition_matrix(mesh,vec{u});
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
  
  mat costs = di.get_costs(points);
  vec weights = lp_norm_weights(points,2);

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

  DoubleIntegratorSimulator di = build_di_simulator(var_map);
  
  TriMesh mesh;
  if(var_map.count("mesh_file")){
    string mesh_file = var_map["mesh_file"].as<string>();
    cout << "Reading in mesh from [" << mesh_file << "]..." << endl;
    mesh.read_cgal(mesh_file);
  }
  else{
    cout << "Generating initial mesh..." << endl;
    generate_initial_mesh(var_map,di,mesh);
  }
  mesh.freeze();

  mat bbox = di.get_bounding_box();
  
  cout << "Mesh stats:" << endl;
  cout << "\tNumber of vertices: " << mesh.number_of_vertices() << endl;
  cout << "\tNumber of faces: " << mesh.number_of_faces() << endl;
  cout << "\tLower bound:" << bbox.col(0).t();
  cout << "\tUpper bound:" << bbox.col(1).t();

  build_lcp(var_map,di,mesh);
}
