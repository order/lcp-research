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

////////////////////////////////////////
// Generate the LCP

void build_minop_lcp(const TriMesh &mesh,
                     const vec & a,
                     LCP & lcp,
                     vec & ans){
  double off = 1.0; // +ve offset
  Points points = mesh.get_spatial_nodes();
  uint N = points.n_rows;
  vec sq_dist = sum(pow(points,2),1);

  vec b = sq_dist + off;
  vec c = max(zeros<vec>(N),1 - sq_dist) + off;
  
  ans = arma::max(zeros<vec>(N), arma::min(b,c));
  
  assert(a.n_elem == b.n_elem);
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
  uint num_flow_basis = 10;
  Points points = mesh.get_spatial_nodes();
  mat value_basis = make_radial_fourier_basis(points,
                                              num_value_basis,
                                              (double)num_value_basis);
  value_basis = orth(value_basis);
  sp_mat sp_value_basis = sp_mat(value_basis);
  
  Points centers = 2 * randu(10,2) - 1;
  mat flow_basis = make_voronoi_basis(points,
                                      centers);
  flow_basis = orth(flow_basis);
  sp_mat sp_flow_basis = sp_mat(flow_basis);
                                         
  vec ref_weights = ones<vec>(N) / (double)N;
  LCP ref_lcp;
  vec ans;
  build_minop_lcp(mesh,ref_weights,ref_lcp,ans);

  block_sp_vec D = {sp_value_basis,
                    sp_flow_basis,
                    sp_flow_basis};  
  sp_mat P = block_diag(D);
  sp_mat U = P.t() * (ref_lcp.M + 1e-8 * speye(3*N,3*N));
  vec q =  P *(P.t() * ref_lcp.q);

  PLCP ref_plcp = PLCP(P,U,q);
  ProjectiveSolver psolver;
  psolver.comp_thresh = 1e-8;
  psolver.max_iter = 250;
  psolver.regularizer = 1e-8;
  psolver.verbose = false;

  cout << "Reference solve..." << endl;
  SolverResult ref_sol = psolver.aug_solve(ref_plcp);
  cout << "\tDone." << endl;
  
  psolver.comp_thresh = 1e-6;
  // Exactish
  vec twiddle = vec(N);
  for(uint i = 0; i < N; i++){
    cout << "Component: " << i << endl;
    LCP twiddle_lcp;
    vec et = zeros<vec>(N);
    et(i) += 1.0 / (double) N;
    assert(size(ref_weights) == size(et));
    build_minop_lcp(mesh,ref_weights + et,twiddle_lcp,ans);
    vec twiddle_q =  P *(P.t() * twiddle_lcp.q);

    PLCP twiddle_plcp = PLCP(P,U,twiddle_q);
    /*vec x = ref_sol.p.head(3*N);
    vec y = ref_sol.d.head(3*N);
    y(i) += 1.0 / (double) N;*/
    SolverResult twiddle_sol = psolver.aug_solve(twiddle_plcp);
                                                 
    twiddle(i) = twiddle_sol.p(i) - ref_sol.p(i);
  }

  uint R = 75;
  mat jitter = mat(N,R);
  mat noise = mat(N,R);
  for(uint i = 0; i < R; i++){
    cout << "Jitter round: " << i << endl;
    LCP jitter_lcp;
    noise.col(i) = max(1e-4*ones<vec>(N), 0.075 * randn<vec>(N));
    build_minop_lcp(mesh,ref_weights + noise.col(i),jitter_lcp,ans);
    vec jitter_q =  P *(P.t() * jitter_lcp.q);

    PLCP jitter_plcp = PLCP(P,U,jitter_q);
    /*vec x = ref_sol.p.head(3*N);
    vec y = ref_sol.d.head(3*N);
    y(i) += 1.0 / (double) N;*/
    SolverResult jitter_sol = psolver.aug_solve(jitter_plcp);
                                                 
    jitter.col(i) = jitter_sol.p.head(N); 
  }
  
  Archiver arch;
  arch.add_vec("p",ref_sol.p.head(N));
  arch.add_vec("twiddle",twiddle);
  arch.add_mat("jitter",jitter);
  arch.add_mat("noise",noise);
  arch.add_sp_mat("flow_basis",sp_flow_basis);
  arch.write(file_base + ".sens");
  
}
