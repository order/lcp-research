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
  
  //cout << "Optimizing (25 rounds of Lloyd)..."<< endl;
  //mesh.lloyd(15);
  mesh.freeze();

  // Write initial mesh to file
  cout << "Writing:"
       << "\n\t" << (filename + ".node") << " (Shewchuk node file)"
       << "\n\t" << (filename + ".ele") << " (Shewchuk element file)"
       << "\n\t" << (filename + ".tri") << " (CGAL mesh file)" << endl;
  mesh.write_shewchuk(filename);
  mesh.write_cgal(filename + ".tri");
}

////////////////////////////////////////
// Generate the LCP

void build_minop_lcp(const TriMesh &mesh,
                     const vec & a,
                     LCP & lcp,
                     vec & ans){
  double off = 0; // +ve offset
  Points points = mesh.get_spatial_nodes();
  uint N = points.n_rows;
  vec sq_dist = sum(pow(points,2),1);

  vec b = sq_dist + off;
  vec c = max(zeros<vec>(N),1 - sq_dist) + off;
  
  ans = arma::max(zeros<vec>(N), arma::min(b,c));
  
  assert(a.n_elem == b.n_elem);
  assert(all(a >= 0));
  vec q = join_vert(-a,
                    join_vert(b,c));
  assert(3*N == q.n_elem);
  assert(not all(q >= 0));
 
  vector<sp_mat> E;
  E.push_back(speye(N,N));
  E.push_back(speye(N,N));
  sp_mat M = build_M(E);
  assert(M.is_square());
  assert(3*N == M.n_rows);

  lcp = LCP(M,q);
}


po::variables_map read_command_line(uint argc, char** argv){
  po::options_description desc("Meshing options");
  desc.add_options()
    ("help", "produce help message")
    ("outfile_base,o", po::value<string>()->required(),
     "Output experimental result file")
    ("edge_length,e", po::value<double>()->default_value(0.05),
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
  generate_minop_mesh(mesh,file_base,edge_length);
  mesh.freeze();

  // Stats
  uint N = mesh.number_of_vertices();
  uint F = mesh.number_of_faces();
  cout << "Mesh stats:"
       << "\n\tNumber of vertices: " << N
       << "\n\tNumber of faces: " << F
       << endl;

  // Build value basis
  uint num_value_basis = 25;
  uint num_flow_basis = 30;
  Points points = mesh.get_spatial_nodes();
  cout << "Generating Radial Fourier basis for value..." << endl;
  mat value_basis = make_radial_fourier_basis(points,
                                              num_value_basis,
                                              (double)num_value_basis);
  cout << "\tOrthogonalizing..." << endl;
  value_basis = orth(value_basis);

  cout << "Building LCP..." << endl;
  vec weights = ones<vec>(N) / (double)N;
  LCP lcp;
  vec ans;
  build_minop_lcp(mesh,weights,lcp,ans);
  assert(N == ans.n_elem);

  ProjectiveSolver psolver;
  psolver.comp_thresh = 1e-12;
  psolver.max_iter = 250;
  psolver.aug_rel_scale = 5;
  psolver.verbose = false;
  psolver.initial_sigma = 0.3;

  double regularizer = 1e-12;

  bvec free_vars = zeros<bvec>(3*N);
  free_vars.head(N).fill(1);

  uint Runs = 300;
  mat residual = mat(Runs,2);
  umat iterations = umat(Runs,2);
  for(uint i = 0; i < Runs; i++){
    cout << "---Iteration " << i << "---" << endl;
    cout << "Generating Voronoi basis for flow..." << endl;
    sp_mat sp_value_basis = sp_mat(value_basis);
  
    Points centers = 2 * randu(10,2) - 1;
    mat flow_basis = make_voronoi_basis(points,
                                        centers);
    cout << "\tOrthogonalizing..." << endl;
    flow_basis = orth(flow_basis);
    sp_mat sp_flow_basis = sp_mat(flow_basis);

    cout << "Building PLCPs..." << endl;
    block_sp_vec D = {sp_value_basis,
                      sp_flow_basis,
                      sp_flow_basis};  
    sp_mat P = block_diag(D);
    sp_mat U = P.t() * (lcp.M + regularizer*speye(size(lcp.M)));
    vec q =  P *(P.t() * lcp.q);

    cout << "Solving free LCP..." << endl;
    PLCP free_plcp = PLCP(P,U,q,free_vars);
    SolverResult free_sol = psolver.aug_solve(free_plcp);
    double free_res = norm(ans - free_sol.p.head(N));
    cout << "\tResidual: " << free_res << endl;
    cout << "\tIteration: " << free_sol.iter << endl;

    cout << "Solving bound LCP..." << endl;
    PLCP bound_plcp = PLCP(P,U,q);
    SolverResult bound_sol = psolver.aug_solve(bound_plcp);
    double bound_res = norm(ans - bound_sol.p.head(N));
    cout << "\tResidual: " << bound_res << endl;
    cout << "\tIteration: " << bound_sol.iter << endl;

    residual.row(i) = rowvec{free_res,bound_res};
    iterations.row(i) = urowvec{free_sol.iter,bound_sol.iter};
  }
  
  Archiver arch;
  arch.add_mat("residual",residual);
  arch.add_umat("iterations",iterations);
  arch.write(file_base + ".exp_res");
  
}
