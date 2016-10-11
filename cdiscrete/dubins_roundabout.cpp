#include "dubins.h"
#include "misc.h"
#include "io.h"
#include "lcp.h"
#include "tet_mesh.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace tet_mesh;
using namespace dubins;

po::variables_map read_command_line(uint argc, char** argv){
  po::options_description desc("Meshing options");
  desc.add_options()
    ("help", "produce help message")
    ("mesh,m", po::value<string>()->required(),
     "Input (CGAL) mesh file");
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
  mat bounds = mesh.find_bounding_box();
  vec lb = bounds.col(0);
  vec ub = bounds.col(1);
  cout << "\tLower bound:" << lb.t();
  cout << "\tUpper bound:" << ub.t();


  RoundaboutDubinsCarSimulator dubins = RoundaboutDubinsCarSimulator(DUBINS_ACTIONS);
  bool include_oob = true;
  LCP L = build_lcp(&dubins,
                    &mesh,
                    DUBINS_GAMMA,
                    include_oob);
  //string filename = var_map["lcp"].as<string>();
  //L.write(filename);

  string filename = "/home/epz/data/dubins.lcp";
  L.write(filename);
}
