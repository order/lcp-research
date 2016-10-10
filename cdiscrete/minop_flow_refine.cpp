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

#define MIN_FLOW_BASIS 10
#define NUM_VALUE_BASIS 25


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
    ("jitter,j", po::value<uint>()->default_value(5),
     "Number of jitter rounds")
    ("refine,r", po::value<uint>()->default_value(5),
     "Number of refinement rounds")
    ("experiment_runs,E", po::value<uint>()->default_value(1),
     "Number of experiment runs")
    ("edge_length,e", po::value<double>()->default_value(0.125),
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

/*
Run one instance (one initial random basis) of flow refinement
based on "jittering": perturbing the state weights and observing
resulting correlation with solution perturbation
*/
vec jitter_experiment_run(const TriMesh & mesh,
                          const po::variables_map var_map){
  // Fixed value basis size and initial basis size
  
  // How many refinement rounds to run
  uint refine_rounds = var_map["refine"].as<uint>();

  // How many jitter programs to solve to get correlation
  uint jitter_rounds = var_map["jitter"].as<uint>();

  // Build value basis
  Points points = mesh.get_spatial_nodes();
  mat value_basis = make_radial_fourier_basis(points,
                                              NUM_VALUE_BASIS,
                                              (double)NUM_VALUE_BASIS);
  value_basis = orth(value_basis); // Orthonormalize (expansive)
  sp_mat sp_value_basis = sp_mat(value_basis);

  // Build initial flow_basis
  Points centers = 2 * randu(MIN_FLOW_BASIS,2) - 1;
  VoronoiBasis flow_basis = VoronoiBasis(points,centers);             

  // Set up solver parameters
  ProjectiveSolver psolver;
  psolver.comp_thresh = 1e-8;
  psolver.max_iter = 500;
  psolver.aug_rel_scale = 5;
  psolver.regularizer = 1e-8;
  psolver.verbose = false;
  psolver.initial_sigma = 0.8;

  // Regularization of LCP problem
  double regularizer = 1e-12;

  // Set up free variables
  uint N = mesh.number_of_vertices();
  bvec free_vars = zeros<bvec>(3*N);
  if(not var_map["bound"].as<bool>())
    free_vars.head(N).fill(1);

  // Build the reference LCP
  vec ref_weights = ones<vec>(N) / (double)N;
  LCP ref_lcp;
  vec ans;
  build_minop_lcp(mesh,ref_weights,ref_lcp,ans);

  // Run the iterations
  vec residual = vec(refine_rounds);
  for(uint I = 0; I < refine_rounds; I++){
    cout << "---Iteration: " << I << "---" << endl;

    cout << "Forming new flow basis..." << endl;
    sp_mat sp_flow_basis = flow_basis.get_basis();
    block_sp_vec D = {sp_value_basis,
                      sp_flow_basis,
                      sp_flow_basis};

    // Build the projective matrices
    sp_mat P = block_diag(D);
    sp_mat U = P.t() * (ref_lcp.M + regularizer * speye(3*N,3*N));
    vec q =  P *(P.t() * ref_lcp.q);
    PLCP ref_plcp = PLCP(P,U,q,free_vars);

    cout << "Reference solve..." << endl;
    SolverResult ref_sol = psolver.aug_solve(ref_plcp);
    double res = norm(ans - ref_sol.p.head(N));
    residual(I) = res;
    cout << "\tResidual:" << res << endl;

    // Add a new random center
    vec new_center = 2*randu<vec>(2) - 1;
    flow_basis.add_center(new_center);

    // If we're not jittering, just use this
    if(0 == jitter_rounds){
      while(true){
        // Use a perturbed version of the selected node
        new_center = 2*randu<vec>(2) - 1;
        flow_basis.replace_last_center(new_center);
        if(0 < flow_basis.min_count())
          break;
        cout << "\tResampling Voronoi center..." << endl;
      }
      continue;     
    }
    
    // Perturb objective function, use correlation between
    // Modified objective value and objective noise
    cout << "Jittering..." <<endl;
    mat jitter = mat(N,jitter_rounds);
    mat noise = mat(N,jitter_rounds);
    jitter_solve(mesh,psolver,ref_plcp,ref_weights,
                 jitter,noise,jitter_rounds);
      
    // Subtract off reference solution
    jitter = (jitter.each_col() - ref_sol.p.head(N));

    // Calculate correlation constant
    vec rho = pearson_rho(jitter,noise);

    // Refine the highest absolute rho node
    uint refine_node = abs(rho).index_max();

    // Make sure we're covering at least one node
    while(true){
      // Use a perturbed version of the selected node
      new_center = points.row(refine_node).t() + 0.5 * randn(2);
      flow_basis.replace_last_center(new_center);
      if(0 < flow_basis.min_count())
        break;
      cout << "\tResampling Voronoi center..." << endl;
    } 
  }
  return residual;
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

  uint runs = var_map["experiment_runs"].as<uint>();
  uint refine_rounds = var_map["refine"].as<uint>();

  mat residuals = mat(runs,refine_rounds);
  for(uint i = 0; i < runs;++i){
    vec run_residual= jitter_experiment_run(mesh,
                                            var_map);
    assert(refine_rounds == run_residual.n_elem);
    residuals.row(i) = run_residual.t();
  }

  Archiver arch;
  arch.add_uvec("num_basis",regspace<uvec>(MIN_FLOW_BASIS,
                                           MIN_FLOW_BASIS + refine_rounds));
  arch.add_mat("residuals",residuals);
  cout << "WRITING TO FILE " << file_base << ".exp_res" << endl;
  arch.write(file_base + ".exp_res");
}
