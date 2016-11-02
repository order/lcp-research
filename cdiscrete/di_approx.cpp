#include "di.h"
#include "misc.h"
#include "io.h"
#include "lcp.h"
#include "solver.h"
#include "basis.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace tri_mesh;

sp_mat make_value_basis(const vec & points){
  uint N = points.n_elem;

  uint k = 15;
  uint K = k+1;

  vec centers = linspace(0,1,k);
  
  double bw = 60;
  mat basis = mat(N,K);
  uint I = 0;
  basis.col(I++) = ones<vec>(N);
  for(uint w = 0; w < k; w++){
    basis.col(I++) = gaussian(points,centers(w),bw);
  }
  assert(I == K);
  return sp_mat(orth(basis));
}

mat build_bbox(){
  double B = 5;
  mat bbox = mat(2,2);
  bbox.col(0).fill(-B);
  bbox.col(1).fill(B);
}

TriMesh generate_initial_mesh(){
  double angle = 0.125;
  double length = 0.1;
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
  uint N = mesh.number_of_vertices();
  assert(N > 0);
  
  DoubleIntegratorSimulator di = build_di_simulator();
  uint A = di.num_actions;
  assert(A >= 2);
  
  double gamma = 0.99;

  // Reference blocks
  vector<sp_mat> blocks = di.lcp_blocks(&mesh,gamma);
  assert(A == blocks.size());
  assert(size(N,N) == blocks.at(0));
  
  double bandwidth = 15;
  double thresh = 1e-3;  
  sp_mat smoother = gaussian_smoother(points,bandwidth,thresh);
  assert(size(N,N) == size(smoother));
  vector<sp_mat> sblocks = block_mult(smoother,blocks);

  // Build and pertrub the q
  mat Q = di.q_mat(&mesh);
  assert(size(N,A+1) == size(Q));
  // TODO: perturb the operator
  
  bvec free_vars = zeros<bvec>((A+1)*N);
  free_vars.head(N).fill(1);

  LCP rlcp = LCP(build_M(blocks),vectorise(Q),free_vars);

  // Solve the problem
  KojimaSolver solver = KojimaSolver();
  SolverResult rsol = solver.aug_solve(rlcp);

  // Build the PLCP problem

  // Record the solution and problem data
  Archiver arch = Archiver();
  arch.add_vec("ref_p",rsol.p);
  arch.add_vec("ref_d",rsol.d);
  arch.write("test.sol");
}
