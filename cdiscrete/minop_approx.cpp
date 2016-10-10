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
#include "minop.h"


#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;
using namespace arma;
using namespace tri_mesh;


po::variables_map read_command_line(uint argc, char** argv){
  po::options_description desc("Meshing options");
  desc.add_options()
    ("help", "produce help message")
    ("outfile_base,o", po::value<string>()->required(),
     "Output experimental result file")
    ("mode,m", po::value<string>()->default_value("voronoi"),
     "Basis mode")
    ("params,p",po::value<vector<double> >()->multitoken(),
     "Parameters for the basis mode")
    ("bound,b", po::value<bool>()->default_value(false),
     "Value variables non-negatively bound")
    ("num_val,v",po::value<uint>()->required(),
     "Number of value bases to use")
    ("num_flow,f",po::value<uint>()->required(),
     "Number of flow bases to use")    
    ("edge_length,e", po::value<double>()->default_value(0.1),
     "Max length of triangle edge");
  
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
  double edge_length = var_map["edge_length"].as<double>();
  double angle = 0.125;
  generate_minop_mesh(mesh,file_base,edge_length,angle);
  mesh.freeze();

  // Stats
  uint V = mesh.number_of_vertices();
  uint F = mesh.number_of_faces();
  cout << "Mesh stats:"
       << "\n\tNumber of vertices: " << V
       << "\n\tNumber of faces: " << F
       << endl;

  // Build LCP
  LCP lcp;
  vec ans;
  vec weights =  ones<vec>(V) / (double)V;
  build_minop_lcp(mesh,weights,lcp,ans);

  // Build value basis
  uint num_value = var_map["num_val"].as<uint>();
  Points points = mesh.get_spatial_nodes();
  uint N = points.n_rows;
  sp_mat value_basis = make_radial_fourier_basis(points,
                                              num_value,
                                              (double)num_value);
  uint num_flow = var_map["num_flow"].as<uint>();
  string mode = var_map["mode"].as<string>();
  vector<double> params;
  if(var_map.count("params") > 0){
    params = var_map["params"].as<vector<double> >();
  }
  assert("voronoi" == mode);
  //sp_mat flow_basis = make_basis(mode,params,points,num_flow);
  Points centers = 2 * randu(num_flow,2) - 1;
  VoronoiBasis voronoi = VoronoiBasis(points,centers);  
  sp_mat flow_basis = voronoi.get_basis();
  
  block_sp_vec D = {value_basis,
                    flow_basis,
                    flow_basis};  
  sp_mat P = block_diag(D);
  sp_mat U = P.t() * (lcp.M + 1e-8 * speye(3*V,3*V));
  vec q =  P *(P.t() * lcp.q);

  bvec free_vars = zeros<bvec>(3*N);
  if(not var_map["bound"].as<bool>())
    free_vars.head(N).fill(1);
  
  PLCP plcp = PLCP(P,U,q,free_vars);
  
  ProjectiveSolver psolver;
  psolver.comp_thresh = 1e-8;
  psolver.max_iter = 500;
  psolver.verbose = false;
      
  SolverResult sol = psolver.aug_solve(plcp);
  double res = norm(sol.p.head(N) - ans);
  cout << "Residual: " << res << endl;
  
  vec data = vec(4 + params.size());
  data(0) = num_value;
  data(1) = num_flow;
  data(2) = res;
  data(3) = sol.iter;  

  // Write results out.
  Archiver arch;
  arch.add_vec("data", data);
  arch.add_mat("centers",centers);
  arch.write(file_base + ".exp_res");
}
