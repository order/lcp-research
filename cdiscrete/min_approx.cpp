#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>
#include <armadillo>

//#include <Eigen/Sparse>

#include "misc.h"
#include "io.h"
#include "tri_mesh.h"
#include "refine.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;
using namespace arma;
using namespace tri_mesh;

/*
typedef Eigen::SparseMatrix<double> EigenSpMat;
typedef Eigen::Triplet<double> EigenTriplet;

EigenSpMat convert(const sp_mat & arma_sp_mat){
  typedef sp_mat::const_iterator iter;
  vector<EigenTriplet> trips;
  for(iter it = arma_sp_mat.begin();
      it != arma_sp_mat.end(); it++){
    if(*it < ALMOST_ZERO) continue;
    trips.push_back(EigenTriplet(it.row(),
                                it.col(),
                                *it));
  }
  
  EigenSpMat A(arma_sp_mat.n_rows,
               arma_sp_mat.n_cols);
  A.setFromTriplets(trips.begin(),trips.end());
  return A;
}
*/

po::variables_map read_command_line(uint argc, char** argv){
  po::options_description desc("Meshing options");
  desc.add_options()
    ("help", "produce help message")
    ("infile_base,i", po::value<string>()->required(), "Input file base")
    ("outfile_base,o", po::value<string>()->required(),"Output plcp file base");
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

  string mesh_file = var_map["infile_base"].as<string>() + ".tri";
  // Read in the CGAL mesh
  TriMesh mesh;
  cout << "Reading in CGAL mesh file [" << mesh_file << "]..."  << endl;
  mesh.read_cgal(mesh_file);
  mesh.freeze();
  uint V = mesh.number_of_vertices();
  uint N = mesh.number_of_all_nodes();
  assert(N == V+1);
  uint F = mesh.number_of_faces();
  cout << "Mesh stats:"
       << "\n\tNumber of vertices: " << V
       << "\n\tNumber of faces: " << F
       << endl;
  Points points = mesh.get_spatial_nodes();

  // Find boundary from the mesh and create the simulator object
  mat bbox = mesh.find_bounding_box();
  vec lb = bbox.col(0);
  vec ub = bbox.col(1);
  cout << "\tLower bound:" << lb.t()
       << "\tUpper bound:" << ub.t();

  string lcp_file = var_map["infile_base"].as<string>() + ".lcp";
  cout << "Reading in lcp file [" << lcp_file << "]..." << endl;
  Unarchiver lcp_unarch = Unarchiver(lcp_file);
  sp_mat M = lcp_unarch.load_sp_mat("M");
  mat q = lcp_unarch.load_vec("q");
  assert(0 == q.n_elem % V);
  assert(q.n_elem == V*3);
  mat q_blocks = reshape(q,V,3); 
  

  double boundary = 1.0 / sqrt(2.0);
  vec sq_dist = sum(pow(points,2),1);

  uint S = 10;
  mat basis = randu<mat>(V,6+S);
  for(uint i = 0; i < 3; i++){
    basis.col(2*i) = q_blocks.col(i);
    basis(find(sq_dist > boundary),uvec{i}).fill(0);
    basis.col(2*i+1) = q_blocks.col(i);
    basis(find(sq_dist <= boundary),uvec{i}).fill(0);
  } 

  mat Q,R;
  cout << "Running QR decomposition (dense; expensive)..." << endl;
  qr_econ(Q,R,basis);
  assert(size(Q) == size(basis));
  
  Q(find(Q < 1e-6)).fill(0);
  sp_mat sp_Q(Q);
  
  block_sp_row D;
  D.push_back(sp_Q);
  D.push_back(speye(V,V));
  D.push_back(speye(V,V));
  sp_mat Phi = diags(D);
  sp_mat U = Phi.t() * M;

  vec r =  Phi.t() * q;

  string plcp_file = var_map["outfile_base"].as<string>() + ".plcp";
  cout << "Writing to " << plcp_file << endl;
  Archiver arch;
  arch.add_sp_mat("Phi",Phi);
  arch.add_sp_mat("U",U);
  arch.add_vec("r",r);
  arch.write(plcp_file);
}
