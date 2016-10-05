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


////////////////////////////////////////
// Generate the LCP

po::variables_map read_command_line(uint argc, char** argv){
  po::options_description desc("Meshing options");
  desc.add_options()
    ("help", "produce help message")
    ("outfile_base,o", po::value<string>()->required(),
     "Output experimental result file")
    ("bound,b", po::value<bool>()->default_value(false),
     "Value variables non-negatively bound")
    ("jitter,j" po::value<uint>()->default_value(25),
     "Number of jitter rounds")
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

vec jitter_experiment_run(const TriMesh & mesh,
                          const po::variables_map var_map){
  uint num_value_basis = 25;
  uint num_flow_basis = 10;
  uint max_iter =  var_map["max_iter"].as<uint>();
  uint max_basis = var_map["max_basis"].as<uint>();
  uint N = mesh.number_of_vertices();
  uint jitter_rounds = var_map["jitter"].as<uint>();

  // Build value basis
  Points points = mesh.get_spatial_nodes();
  mat value_basis = make_radial_fourier_basis(points,
                                              num_value_basis,
                                              (double)num_value_basis);
  value_basis = orth(value_basis);
  sp_mat sp_value_basis = sp_mat(value_basis);

  // Build initial flow_basis
  Points centers = 2 * randu(num_flow_basis,2) - 1;
  VoronoiBasis flow_basis = VoronoiBasis(points,centers);             

  // Solver set up
  ProjectiveSolver psolver;
  psolver.comp_thresh = 1e-8;
  psolver.max_iter = 250;
  psolver.aug_rel_scale = 5;
  psolver.regularizer = 1e-8;
  psolver.verbose = false;
  psolver.initial_sigma = 0.3;

  double regularizer = 1e-12;

  // Free variables
  bvec free_vars = zeros<bvec>(3*N);
  if(not var_map["bound"].as<bool>())
    free_vars.head(N).fill(1);

  // Reference LCP
  vec ref_weights = ones<vec>(N) / (double)N;
  LCP ref_lcp;
  vec ans;
  build_minop_lcp(mesh,ref_weights,ref_lcp,ans);

  vec residual = vec(max_iter);
  for(uint I = 0; I < max_iter; I++){
    cout << "---Iteration: " << I << "---" endl;

    cout << "Forming new flow basis..." << endl;
    sp_mat sp_flow_basis = flow_basis.get_basis();
    block_sp_vec D = {sp_value_basis,
                      sp_flow_basis,
                      sp_flow_basis};
    vec rand_basis_vec = sp_flow_basis * rand(flow_basis.n_basis);
    cout << "\tWriting random vector in span to file..." << endl;
    
    sp_mat P = block_diag(D);
    sp_mat U = P.t() * (ref_lcp.M + regularizer * speye(3*N,3*N));
    vec q =  P *(P.t() * ref_lcp.q);

    PLCP ref_plcp = PLCP(P,U,q,free_vars);
    cout << "Reference solve..." << endl;
    SolverResult ref_sol = psolver.aug_solve(ref_plcp);
    double res = norm(ans - ref_sol.p.head(N));
    cout << "\tResidual:" << res << endl;

    // Add a new center
    vec new_center = 2*randu<vec>(2) - 1;
    flow_basis.add_center(new_center);
    if(jitter_rounds){
      // Perturb objective function, use correlation between
      // Modified objective value and objective noise
      cout << "Jittering..." <<endl;
      mat jitter = mat(N,jitter_rounds);
      mat noise = mat(N,jitter_rounds);
      jitter_solve(mesh,ref_plcp,ref_weights,
                   jitter, noise, jitter_rounds);
      
      // Subtract off reference solution
      jitter = (jitter.each_col() - ref_sol.head(N));

      // Calculate correlation
      vec rho = pearson_rho(jitter,noise);

      // Refine the highest absolute rho node
      uint refine_node = abs(rho).index_max();
      while(true){
        // Use a perturbed version of the selected node
        new_center = points.row(refine_node).t() + 0.5 * randn(2);
        flow_basis.replace_last_center(new_center);
        if(0 < flow_basis.count(n_basis-1))
          break; // Covers at least one node.
        cout << "\tResampling Voronoi center..." << endl;
      } 
    }
    else
      cout << "Using random center." << endl;
  }
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
  generate_minop_mesh(mesh,file_base,edge_length);
  mesh.freeze();

  uint max_iter =  var_map["max_iter"].as<uint>();
  uint min_basis = var_map["max_basis"].as<uint>();
  
  mat residuals = mat(max_iter,max_basis);
  for(uint I = 0; I < max_iter;++I){
    vec run_residual= jitter_experiment_run(mesh,
                                            var_map);
    residuals.row(i) = run_residual;
  }
  Archiver arch;
  arch.add_mat("residuals",residuals);
  arch.write(file_base + ".exp_res");
}
