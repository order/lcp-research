#include "misc.h"
#include "io.h"
#include "lcp.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "tri_mesh.h"
using namespace tri_mesh;

vec besselish_function(const Points & points,
                       const vec & center,
                       const double b){
  uint N = points.n_rows;
  assert(center.n_elem == points.n_cols);
  vec d = lp_norm(points - repmat(center.t(),N,1),2,1);
  return cos(b*b*d) % exp(-b*pow(d,2));
}

void generate_initial_mesh(const po::variables_map & var_map,
                           TriMesh & mesh){
  double angle = var_map["mesh_angle"].as<double>();
  double length = var_map["mesh_length"].as<double>();

  cout << "Initial meshing..."<< endl;
  mesh.build_box_boundary({{-1,1},{-1,1}});
  
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

LCP build_min_op_lcp(const TriMesh &mesh){
  uint N = mesh.number_of_spatial_nodes();
  Points points = mesh.get_spatial_nodes();

  vec a = ones<vec>(N) / (double) N;
  vec sq_dist = sum(pow(points,2),1);
  vec b = sq_dist;
  vec c = max(zeros<vec>(N),1 - sq_dist);
  
  assert(a.n_elem == b.n_elem);
  vec q = join_vert(-a,join_vert(b,c));
  assert(3*N == q.n_elem);
 
  vector<sp_mat> E;
  E.push_back(speye(N,N));
  E.push_back(speye(N,N));
  sp_mat M = build_M(E);
  assert(M.is_square());
  assert(3*N == M.n_rows);

  return LCP(M,q);
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
    ("mesh_length", po::value<double>()->default_value(0.2),
     "Mesh edge length refinement criterion");
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

  mat bbox = mesh.find_bounding_box();
  cout << "Mesh stats:" << endl;
  cout << "\tNumber of vertices: " << mesh.number_of_vertices() << endl;
  cout << "\tNumber of faces: " << mesh.number_of_faces() << endl;
  cout << "\tLower bound:" << bbox.col(0).t();
  cout << "\tUpper bound:" << bbox.col(1).t();

  bool include_oob = true;
  
  LCP L = build_min_op_lcp(mesh);
  string filename = var_map["outfile_base"].as<string>() + ".lcp";
  cout << "Writing LCP file " << filename << "..." << endl;
  L.write(filename);
}
