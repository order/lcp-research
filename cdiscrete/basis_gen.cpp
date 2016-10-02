#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>
#include <armadillo>

//#include <Eigen/Sparse>

#include "misc.h"
#include "io.h"
#include "tri_mesh.h"
#include "lcp.h"
#include "basis.h"
#include "solver.h"


#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;
using namespace arma;
using namespace tri_mesh;

void generate_minop_mesh(TriMesh & mesh,
                         const string & filename,
                         double edge_length){
  double angle = 0.125;
  cout << "Initial meshing..."<< endl;
  mesh.build_box_boundary({{-1.1,1.1},{-1.1,1.1}});
  mesh.build_circle(zeros<vec>(2),50,1.0);
  mesh.build_circle(zeros<vec>(2),30,1.0/sqrt(2.0));
  mesh.build_circle(zeros<vec>(2),25,0.25);

  cout << "Refining based on (" << angle
       << "," << edge_length <<  ") criterion ..."<< endl;
  mesh.refine(angle,edge_length);
  
  cout << "Optimizing (25 rounds of Lloyd)..."<< endl;
  mesh.lloyd(25);
  mesh.freeze();

  // Write initial mesh to file
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
     "Output experimental result file");
  
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
  arma_rng::set_seed_random();

  // Read in the CGAL mesh
  TriMesh mesh;
  string file_base = var_map["outfile_base"].as<string>();

  generate_minop_mesh(mesh,file_base,0.2);
  mesh.freeze();

  // Stats
  uint N = mesh.number_of_vertices();
  uint F = mesh.number_of_faces();
  cout << "Mesh stats:"
       << "\n\tNumber of vertices: " << N
       << "\n\tNumber of faces: " << F
       << endl;

  // Build value basis
  Points points = mesh.get_spatial_nodes();
  Points centers = 2 * randu(5,2) - 1;
  IndexPartition partition = voronoi_partition(points,
                                               centers);
  set<uint> ball = ball_indices(points, zeros<vec>(2), 7);
  add_basis(partition,ball);
  sp_mat basis = build_basis_from_partition(partition, N);

  
  uint K = basis.n_cols;
  Archiver arch;

  vec w = randn<vec>(K);
  vec sparse_vector = basis * w;
  cout << size(sparse_vector) << endl;

  arch.add_vec("vector",sparse_vector);

  arch.write(file_base + ".basis");
}
