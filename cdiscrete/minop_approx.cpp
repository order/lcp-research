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
     "Output plcp file base")
    ("mesh,m",po::value<string>(),
     "Optional input mesh (generates otherwise)")
    ("edge_length,e", po::value<double>()->default_value(0.125),
     "Max length of triangle edge")
    ("Fourier,F", po::value<uint>()->required(),"Number of Fourier bases")
    ("Voronoi,V", po::value<uint>()->required(),"Number of Voronoi bases");
  
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


LCP build_minop_lcp(const TriMesh &mesh,
                    const vec & a,
                    const vec & b,
                    const vec & c){
  uint N = mesh.number_of_spatial_nodes();
  Points points = mesh.get_spatial_nodes();
  
  assert(a.n_elem == b.n_elem);
  assert(all(a < 0));
  vec q = join_vert(a,
                    join_vert(b,c));
  assert(3*N == q.n_elem);
  assert(not all(q >= 0));
 
  vector<sp_mat> E;
  E.push_back(speye(N,N));
  E.push_back(speye(N,N));
  sp_mat M = build_M(E);
  assert(M.is_square());
  assert(3*N == M.n_rows);

  return LCP(M,q);
}

int main(int argc, char** argv)
{
  po::variables_map var_map = read_command_line(argc,argv);
  string filename = var_map["outfile_base"].as<string>();
  
  // Read in the CGAL mesh
  TriMesh mesh;
  if(var_map.count("mesh") > 0){
    string meshfile = var_map["mesh"].as<string>();
    mesh.read_cgal(meshfile);
  }
  else{
    double edge_length = var_map["edge_length"].as<double>();
    generate_minop_mesh(mesh,
                        filename,
                        edge_length);
  }
  mesh.freeze();
  
  uint V = mesh.number_of_vertices();
  uint F = mesh.number_of_faces();
  cout << "Mesh stats:"
       << "\n\tNumber of vertices: " << V
       << "\n\tNumber of faces: " << F
       << endl;

  // Find boundary from the mesh and create the simulator object
  mat bbox = mesh.find_bounding_box();

  arma_rng::set_seed_random();
  double off = 1;
  Points points = mesh.get_spatial_nodes();
  vec sq_dist = sum(pow(points,2),1);
  vec noise = randn<vec>(V);
  vec a = abs(noise);
  a /= accu(a);
  vec jitter = (a - mean(a)) / mean(a);
  
  vec b = sq_dist + off;
  vec c = max(zeros<vec>(V),1 - sq_dist) + off;
  
  LCP lcp = build_minop_lcp(mesh,-a,b,c);
  KojimaSolver ksolver;
  cout << "Solving..." << endl;
  pd_pair sol = ksolver.aug_solve(lcp);
  cout << "Done." << endl;
 
  string lcp_file = filename + ".lcp";
  cout << "Writing to " << lcp_file << endl;
  lcp.write(lcp_file);

  return 0;

  uint num_fourier = var_map["Fourier"].as<uint>();
  mat value_basis = make_radial_fourier_basis(points,
                                              num_fourier,
                                              (double)num_fourier);
  value_basis = orth(value_basis);

  arma_rng::set_seed(0);
  uint num_voronoi = var_map["Voronoi"].as<uint>();
  Points centers = 2 * randu(num_voronoi,2) - 1;
  mat flow_basis = make_voronoi_basis(points,
                                      centers);
  flow_basis = orth(flow_basis);

  block_sp_vec D = {sp_mat(value_basis),
                    sp_mat(flow_basis),
                    sp_mat(flow_basis)};
  
  sp_mat Phi = block_diag(D);
  sp_mat U = Phi.t() * (lcp.M + 1e-8 * speye(3*V,3*V));// * Phi * Phi.t();
  vec r =  Phi.t() * lcp.q;

  string plcp_file = filename + ".plcp";
  cout << "Writing to " << plcp_file << endl;
  Archiver plcp_arch;
  plcp_arch.add_sp_mat("Phi",Phi);
  plcp_arch.add_sp_mat("U",U);
  plcp_arch.add_vec("r",r);
  plcp_arch.add_vec("a",a);
  plcp_arch.add_vec("jitter",jitter);  
  plcp_arch.add_vec("b",b);
  plcp_arch.add_vec("c",c);
  plcp_arch.add_mat("value_basis",value_basis);
  plcp_arch.add_mat("flow_basis",flow_basis);  
  plcp_arch.write(plcp_file);
}
