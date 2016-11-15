#include "di.h"
#include "misc.h"
#include "io.h"
#include "lcp.h"
#include "solver.h"
#include "basis.h"
#include "smooth.h"
#include "refine.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace tri_mesh;

#define B 5.0
#define LENGTH 0.3
#define GAMMA 0.995
#define SMOOTH_BW 1e9
#define SMOOTH_THRESH 1e-4


sp_mat make_value_basis(const Points & points,
                        uint rbf_grid_size,
                        double rbf_bandwidth){

  uint N = points.n_rows;
  vec grid = linspace<vec>(-B,B,rbf_grid_size);
  vector<vec> grids;
  grids.push_back(grid);
  grids.push_back(grid);

  mat centers = make_points(grids);
  mat basis = make_rbf_basis(points,centers,rbf_bandwidth,0);
  //sp_mat basis = make_voronoi_basis(points,centers);
  //sp_mat basis = speye(N,N);
  return sp_mat(basis);
}

mat build_bbox(){
  mat bbox = mat(2,2);
  bbox.col(0).fill(-B);
  bbox.col(1).fill(B);
  return bbox;
}

TriMesh generate_initial_mesh(){
  double angle = 0.125;
  double length = LENGTH;
  mat bbox = build_bbox();
  return generate_initial_mesh(angle,length,bbox);
}

DoubleIntegratorSimulator build_di_simulator(){
  mat bbox = build_bbox();
  mat actions = vec{-1,1};
  double noise_std = 0.0;
  double step = 0.025;
  return DoubleIntegratorSimulator(bbox,actions,noise_std,step);
}

////////////////////
// MAIN FUNCTION //
///////////////////

int main(int argc, char** argv)
{
  arma_rng::set_seed_random();
  // Set up 2D space

  cout << "Generating initial mesh..." << endl;
  TriMesh mesh = generate_initial_mesh();
  Points points = mesh.get_spatial_nodes();
  uint N = points.n_rows;
  assert(N == mesh.number_of_spatial_nodes());
  assert(N > 0);
  
  DoubleIntegratorSimulator di = build_di_simulator();
  uint A = di.num_actions();
  assert(A >= 2);
  
  // Reference blocks
  cout << "Building LCP blocks..." << endl;
  vector<sp_mat> blocks = di.lcp_blocks(&mesh,GAMMA);
  assert(A == blocks.size());
  assert(size(N,N) == size(blocks.at(0)));

  // Build smoother
  cout << "Building smoother matrix..." << endl;
  sp_mat smoother = gaussian_smoother(points,SMOOTH_BW,SMOOTH_THRESH);
  assert(size(N,N) == size(smoother));

  // Build and pertrub the q
  cout << "Building RHS Q..." << endl;
  mat Q = di.q_mat(&mesh);
  
  bvec free_vars = zeros<bvec>((A+1)*N);
  free_vars.head(N).fill(1);
  
  ProjectiveSolver psolver = ProjectiveSolver();
  psolver.comp_thresh = 1e-12;
  psolver.initial_sigma = 0.25;
  psolver.verbose = false;
  psolver.iter_verbose = false;

  vec bandwidth = vec{0.4,0.5,0.6,0.75,1.0,1.25};
  uvec grid_size = regspace<uvec>(5,17);

  cube data = cube(bandwidth.n_elem,
                   grid_size.n_elem,
                   3);

  for(uint i = 0; i < bandwidth.n_elem; i++){
    double bw = bandwidth(i);
    cout << "Bandwidth " << bw << endl;
    for(uint j = 0; j < grid_size.n_elem; j++){
      uint G = grid_size(j);
      cout << "\tGrid size " << G << endl;
      sp_mat value_basis = make_value_basis(points,G,bw);
      PLCP plcp = approx_lcp(value_basis,smoother,
                             blocks,Q,free_vars);
      SolverResult sol = psolver.aug_solve(plcp);
      mat P = reshape(sol.p,N,A+1);
      vec V = P.col(0);
      
      vec res = bellman_residual_at_nodes(&mesh,&di,V,GAMMA);
      data(i,j,0) = norm(res,1);
      data(i,j,1) = norm(res,2);
      data(i,j,2) = norm(res,"inf");
      rowvec tube = data.tube(i,j);
      cout<< "\t\tResidual: " << tube;
    }
  }
  
  mesh.write_cgal("test.mesh");
  Archiver arch = Archiver();
  arch.add_cube("data",data);
  arch.add_vec("bandwidth",bandwidth);
  arch.add_uvec("grid_size",grid_size);

  arch.write("grids.dat");
}
