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
                     LCP & lcp,
                     vec & ans){
  double off = 1.0; // +ve offset
  Points points = mesh.get_spatial_nodes();
  uint N = points.n_rows;
  vec sq_dist = sum(pow(points,2),1);

  vec a = ones<vec>(N) / (double)N;
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
    ("mode,m", po::value<string>()->required(),
     "Basis mode")
    ("params,p",po::value<vector<double> >()->multitoken(),
     "Parameters for the basis mode")
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
  uint V = mesh.number_of_vertices();
  uint F = mesh.number_of_faces();
  cout << "Mesh stats:"
       << "\n\tNumber of vertices: " << V
       << "\n\tNumber of faces: " << F
       << endl;

  // Build LCP
  LCP lcp;
  vec ans;
  build_minop_lcp(mesh,lcp,ans);

  // Build value basis
  uint num_fourier = 25;
  Points points = mesh.get_spatial_nodes();
  uint N = points.n_rows;
  mat value_basis = make_radial_fourier_basis(points,
                                              num_fourier,
                                              (double)num_fourier);
  value_basis = orth(value_basis);
  sp_mat sp_value_basis = sp_mat(value_basis);

  uint min_k = 10;
  uint max_k = 150;
  uvec num_basis = regspace<uvec>(min_k,2,max_k);

  uint K = num_basis.n_elem;
  uint R = 5;
  
  mat res_data = mat(K,R);
  mat iter_data = mat(K,R);

  string mode = var_map["mode"].as<string>();
  vector<double> params;
  if(var_map.count("params") > 0){
    params = var_map["params"].as<vector<double> >();
  }
  
  for(uint i = 0; i < K; i++){
    uint k = num_basis(i);
    cout << mode << " basis size: " << k << endl;
    cout << "\tRun:";
    for(uint r = 0; r < R; r++){
      cout << " " << r;
      cout.flush();
      sp_mat sp_flow_basis;
      if(0 == mode.compare("voronoi")){
        assert(0 == params.size());
        Points centers = 2 * randu(k,2) - 1;
        mat flow_basis = make_voronoi_basis(points,
                                            centers);
        flow_basis = orth(flow_basis);
        sp_flow_basis = sp_mat(flow_basis);
      }
      else if (0 == mode.compare("sample")){
        assert(0 == params.size());
        sp_flow_basis = make_sample_basis(N,k);
        mat flow_basis = mat(sp_flow_basis);
        flow_basis = orth(flow_basis);
        sp_flow_basis = sp_mat(flow_basis);
      }
      else if (0 == mode.compare("balls")){
        assert(1 == params.size());
        uint radius = (uint) params[0];
        Points centers = 2 * randu(k,2) - 1;
        sp_flow_basis = make_ball_basis(points,centers,radius);
        mat flow_basis = mat(sp_flow_basis);
        flow_basis = orth(flow_basis);
        sp_flow_basis = sp_mat(flow_basis);
      }
      else if (0 == mode.compare("rbf")){
        assert(1 == params.size());
        double bandwidth = (double) params[0];
        Points centers = 2 * randu(k,2) - 1;
        mat flow_basis = make_rbf_basis(points,centers,bandwidth);
        sp_flow_basis = sp_mat(flow_basis);
      }
      else{
        assert(false);
      }      
      
      block_sp_vec D = {sp_value_basis,
                        sp_flow_basis,
                        sp_flow_basis};  
      sp_mat P = block_diag(D);
      sp_mat U = P.t() * (lcp.M + 1e-8 * speye(3*V,3*V));
      vec q =  P *(P.t() * lcp.q);

      PLCP plcp = PLCP(P,U,q);
      ProjectiveSolver psolver;
      psolver.comp_thresh = 1e-6;
      psolver.max_iter = 250;
      psolver.regularizer = 1e-8;
      psolver.verbose = false;
      
      SolverResult sol = psolver.aug_solve(plcp);
      res_data(i,r) = norm(sol.p.head(N) - ans);
      iter_data(i,r) = sol.iter;
    }
    cout << endl; // For the run print out;
  }
  Archiver arch;
  arch.add_mat("res_data",res_data);
  arch.add_mat("iter_data",iter_data);
  arch.add_uvec("num_basis",num_basis);
  arch.write(file_base + ".exp_res");
}
