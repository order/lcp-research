#include "misc.h"
#include "io.h"
#include "tet_mesh.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

TetMesh read_mesh(const string & mesh_file){
  TetMesh mesh;
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
  return mesh;
}

Points read_points(const string & point_file){
  cout << "Reading in points from [" << point_file << "]" << endl;
  Unarchiver unarch = Unarchiver(point_file);
  Points points = unarch.load_mat("points");
  cout << "\tFound points: " << size(points) << endl;;
  assert(TET_NUM_DIM == points.n_cols);

  return points;
}

vec read_values(const TetMesh & mesh,
                const string & value_file,
                double oob){
  cout << "Reading in values from [" << value_file << "]" << endl;
  Unarchiver unarch = Unarchiver(value_file);
  vec values = unarch.load_vec("values");
  cout << "\tFound values: " << size(values) << endl;;

  if(values.n_elem == mesh.number_of_vertices()){
    cout << "Values only provided for vertices, filling in..." << endl;
    values = join_vert(values,vec{oob});
  }
  assert(values.n_elem == mesh.number_of_nodes());
  
  return values; 
}

void write_interp_to_file(const TetMesh & mesh,
                          const Points & points,
                          const vec & values,
                          const string & out_file){
  cout << "Interpolating..." << endl;
  uint N = points.n_rows;
  ElementDist dist = mesh.points_to_element_dist(points);
  vec interp = dist.t() * values;
  assert(N == interp.n_elem);
  Archiver arch;
  arch.add_vec("interp",interp);
  arch.write(out_file);
  cout << "\tWrote [" << out_file << "]." << endl;
}

po::variables_map read_command_line(uint argc, char** argv){
  po::options_description desc("Meshing options");
  desc.add_options()
    ("help", "produce help message")
    ("points,p", po::value<string>()->required(),
     "Points file")
    ("mesh,m", po::value<string>()->required(),
     "Input (CGAL) mesh file")
    ("values,v", po::value<string>()->required(),
     "Value at nodes file")
    ("out,o", po::value<string>()->required(),
     "File with values at points")
    ("oob,b", po::value<double>()->default_value(datum::nan),
     "OOB fill value");
  po::variables_map var_map;
  po::store(po::parse_command_line(argc, argv, desc), var_map);
  po::notify(var_map);
  if (var_map.count("help")) {
    cout << desc << "\n";
    exit(1);
  }
  return var_map;
}

int main(int argc, char** argv)
{
  po::variables_map var_map = read_command_line(argc,argv);
  TetMesh mesh = read_mesh(var_map["mesh"].as<string>());
  Points points = read_points(var_map["points"].as<string>());
  double oob = var_map["oob"].as<double>();
  vec values = read_values(mesh, var_map["values"].as<string>(),oob);
  write_interp_to_file(mesh, points, values,var_map["out"].as<string>());
}
