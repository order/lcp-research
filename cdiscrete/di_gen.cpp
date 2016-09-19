#include "di.h"
#include "misc.h"
#include "io.h"
#include "lcp.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace tri_mesh;

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

po::variables_map read_command_line(uint argc, char** argv){
  po::options_description desc("Meshing options");
  desc.add_options()
    ("help", "produce help message")
    ("outfile_base,o", po::value<string>()->required(),
     "Base for all files generated (lcp, mesh)")
    ("mesh_file,m", po::value<string>(), "Input (CGAL) mesh file")
    ("mesh_angle", po::value<double>()->default_value(0.125),
     "Mesh angle refinement criterion")
    ("mesh_length", po::value<double>()->default_value(1),
     "Mesh edge length refinement criterion")    
    ("gamma,g", po::value<double>()->default_value(0.997),
     "Discount factor")
    ("boundary,B", po::value<double>()->default_value(5.0),
     "Square boundary box [-B,B]^2")
    ("bang_points,b", po::value<uint>()->default_value(10),
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

  double gamma = var_map["gamma"].as<double>();
  bool include_oob = false;
  LCP L = build_lcp(&di,&mesh,gamma,include_oob);

  string filename = var_map["outfile_base"].as<string>() + ".lcp";
  L.write(filename);
}
