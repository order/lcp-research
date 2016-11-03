#include "di.h"
#include "misc.h"
#include "io.h"
#include "lcp.h"
#include "solver.h"
#include "basis.h"
#include "smooth.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace tri_mesh;

#define B 5.0

sp_mat make_value_basis(const Points & points){

  uint N = points.n_rows;
  
  uint k = 9;
  vec grid = linspace<vec>(-B,B,k);
  vector<vec> grids;
  grids.push_back(grid);
  grids.push_back(grid);

  mat centers = make_points(grids);
  double bandwidth = 50;
  mat basis = make_rbf_basis(points,centers,bandwidth);

  basis = orth(join_horiz(ones<vec>(N),basis));
  
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
  double length = 0.3;
  mat bbox = build_bbox();
  return generate_initial_mesh(angle,length,bbox);
}

DoubleIntegratorSimulator build_di_simulator(){
  mat bbox = build_bbox();
  mat actions = vec{-1,1};
  return DoubleIntegratorSimulator(bbox,actions);
}

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
  
  double gamma = 0.99;

  // Reference blocks
  cout << "Building LCP blocks..." << endl;
  vector<sp_mat> blocks = di.lcp_blocks(&mesh,gamma);
  assert(A == blocks.size());
  assert(size(N,N) == size(blocks.at(0)));
  
  double bandwidth = 75;
  double thresh = 1e-3;  
  sp_mat smoother = gaussian_smoother(points,bandwidth,thresh);
  assert(size(N,N) == size(smoother));
  vector<sp_mat> sblocks = block_mult(smoother,blocks);

  // Build and pertrub the q
  mat Q = di.q_mat(&mesh);
  assert(size(N,A+1) == size(Q));
  assert(all(Q.col(0) <= 0));
  // TODO: perturb the operator
  
  bvec free_vars = zeros<bvec>((A+1)*N);
  free_vars.head(N).fill(1);

  cout << "Assembling blocks into LCP..." << endl;
  cout << "\tBuilding M..." << endl;
  sp_mat M = build_M(sblocks);
  cout << "\tVectorizing Q..." << endl;
  vec q = vectorise(Q);
  cout << "\tBuilding LCP object..." << endl;
  LCP lcp = LCP(M,q,free_vars);

  sp_mat value_basis = make_value_basis(points);
  PLCP plcp = approx_lcp(points,value_basis,blocks,Q,free_vars);
  

  // Solve the problem
  cout << "Initializing Kojima solver..." << endl;
  ProjectiveSolver solver = ProjectiveSolver();
  solver.comp_thresh = 1e-12;
  cout << "Starting augmented LCP solve..."  << endl;
  SolverResult rsol = solver.aug_solve(plcp);

  // Build the PLCP problem

  // Record the solution and problem data
  mesh.write_cgal("test.mesh");
  Archiver arch = Archiver();
  arch.add_vec("p",rsol.p);
  arch.add_vec("d",rsol.d);
  arch.write("test.sol");
}
