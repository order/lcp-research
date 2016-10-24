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

sp_mat make_hand_value_basis(const Points & points){
  uint N = points.n_rows;
  int k = 4;
  int K = 2*(k+1)*(2*k+1) + 1;
  vec d = lp_norm(points,2,1);

  mat basis = mat(N,K);
  basis.col(0) = ones<vec>(N);

  uint I = 1;
  double bandwidth = 4;
  for(int i = -k; i <= k; i++){
    for(int j = 0; j <= k; j++){
      for(int s = 0; s <= 1; s++){
        basis.col(I++) = gabor_wavelet(points,
                                       zeros<vec>(2),
                                       vec{(double)i,(double)j},
                                       bandwidth,0.5*datum::pi*s);
      }
    }
  }

  assert(I == K);
  return sp_mat(orth(basis));
}

sp_mat make_hand_flow_basis(const Points & points){
  uint N = points.n_rows;
  int k = 5;
  int K = 2*(k+1)*(2*k+1) + 1;
  vec d = lp_norm(points,2,1);

  mat basis = zeros<mat>(N,K);
  basis.col(0) = ones<vec>(N);

  uint I = 1;
  double bandwidth = 1;
  for(int i = -k; i <= k; i++){
    for(int j = 0; j <= k; j++){
      for(int s = 0; s <= 1; s++){
        vec wavelet = gabor_wavelet(points,
                                    zeros<vec>(2),
                                    vec{(double)i,(double)j},
                                    bandwidth,0.5*datum::pi*s);
        basis(find(wavelet > -0.05), uvec{I++}).fill(1);
      }
    }
  } 
  assert(I == K);
  
  return sp_mat(orth(basis));
}

vec simple_q(const Points & points){
  uint N = points.n_rows;

  vec center = vec{0,0};
  //vec d = lp_norm(points.each_row() - center.t(),2,1);
  vec d = max(abs(points.each_row() - center.t()),1);
  
  vec weights = ones<vec>(N);
  
  vec bump = rectify(1 - d);
  vec q = vec(3*N);
  q.head(N) = -normalise(weights);
  assert(all(q.head(N) < 0));
  q.subvec(N,size(bump)) = rectify(bump + 0.1*randn(N));
  q.tail(N) = rectify(d + 0.1*randn(N));
  //q.subvec(N,size(bump)) = randu(N);
  //q.tail(N) = randu(N);
  return q;  
}

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
    ("save_sol,s", po::value<bool>()->default_value(false),
     "Save solution to file")
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
  if(var_map["save_sol"].as<bool>()){
    cout << "Writing mesh file " << file_base << ".mesh" << endl;
    mesh.write_cgal(file_base + ".mesh");
  }
  mesh.freeze();

  // Stats
  uint V = mesh.number_of_vertices();
  uint F = mesh.number_of_faces();
  cout << "Mesh stats:"
       << "\n\tNumber of vertices: " << V
       << "\n\tNumber of faces: " << F
       << endl;
  Points points = mesh.get_spatial_nodes();
  uint N = points.n_rows;
  assert(V == N);


  // Build LCP
  vec q = simple_q(points);
  LCP lcp;
  vec ans;
  build_minop_lcp(mesh,q,lcp,ans);


  // Build value basis (Gabor, currently)
  sp_mat value_basis = make_hand_value_basis(points);

  // Project q onto it
  block_sp_vec Dv;
  Dv.push_back(value_basis);
  Dv.push_back(value_basis);
  Dv.push_back(value_basis);
  sp_mat Pv = block_diag(Dv);
  //vec proj_q = Pv * Pv.t() * q; // Project q onto value basis.
  vec proj_q = q;
  
  // Build the smoothed LCP
  double bandwidth = 1e6;
  sp_mat smooth_I;
  LCP smoothed_lcp;
  vec smoothed_ans;
  build_smoothed_minop_lcp(mesh,proj_q,bandwidth,
                                    smoothed_lcp,
                                    smoothed_ans,
                                    smooth_I);
  cout << "Smoothed identity sparsity: " << sparsity(smooth_I) << endl;

  // Build the S&S basis for the smoothed LCP
  mat dense_flow_basis = mat(smooth_I) * make_hand_value_basis(points);
  dense_flow_basis = orth(dense_flow_basis);
  sp_mat flow_basis = sp_mat(dense_flow_basis);

  // Convert smoothed basis into PLCP
  block_sp_vec D;
  D.push_back(value_basis);
  D.push_back(flow_basis);
  D.push_back(flow_basis);
  
  sp_mat P = block_diag(D);
  sp_mat U = P.t() * (smoothed_lcp.M + 1e-10 * speye(3*V,3*V)) * P * P.t();
  vec pq =  P *(P.t() * smoothed_lcp.q);

  bvec free_vars = zeros<bvec>(3*N);
  if(not var_map["bound"].as<bool>()){
    free_vars.head(N).fill(1);
    cout << "Value variables free" << endl;
  }
  else{
    cout << "Value variables non-negative" << endl;
  }

  
  PLCP psmoothed_lcp = PLCP(P,U,pq,free_vars);
  
  ProjectiveSolver psolver;
  psolver.comp_thresh = 1e-15;
  psolver.max_iter = 2500;
  psolver.verbose = true;

  KojimaSolver ksolver;
  ksolver.comp_thresh = 1e-15;
  ksolver.max_iter = 2500;
  ksolver.verbose = true;
      
  SolverResult sol = psolver.aug_solve(psmoothed_lcp);
  //SolverResult sol = ksolver.aug_solve(smoothed_lcp);
  vec res = sol.p.head(N) - ans;
  double res_norm = norm(res);

  cout << "Residual l1-norm: " << norm(res,1) << endl;
  cout << "Residual l2-norm: " << norm(res,2) << endl;
  cout << "Residual linf-norm: " << norm(res,"inf") << endl;

  cout << "Iterations: " << sol.iter << endl;


  // LS reconstruction for reference
  sp_mat LS_mat = value_basis.t() * value_basis;
  vec ls_w = spsolve(LS_mat,value_basis.t()*ans);
  vec ls_recon = value_basis * ls_w;
  
  if(var_map["save_sol"].as<bool>()){
    cout << "Writing solution file " << file_base << ".sol" << endl;
    sol.write(file_base + ".sol");
    Archiver arch;
    arch.add_vec("p", sol.p);
    arch.add_vec("d", sol.d);
    arch.add_vec("ans",ans);
    arch.add_vec("res",res);
    arch.add_vec("ls_recon",ls_recon);
    arch.write(file_base + ".sol");
  }  
}
