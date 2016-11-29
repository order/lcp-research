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
#define LENGTH 0.15
#define GAMMA 0.99

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
  double step = 0.01;
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

  //vec V = laplacian(points,zeros<vec>(2),1);
  vec V = make_ball(points,zeros<vec>(2),1);
  //vec V = gaussian(points,zeros<vec>(2),0.5);
  
  mat basis = mat(N,A+1);

  basis.col(0) = V;
  for(uint i = 0; i < A; i++){
    basis.col(i+1) = -blocks.at(i).t() * V;
  }

  mat ibasis = mat(N,A+1);
  ibasis.col(0) = V;
  for(uint i = 0; i < A; i++){
    sp_mat E = blocks.at(i);
    sp_mat A = E * E.t();
    vec b = E * V;
    ibasis.col(i+1) = spsolve(A,b);
  }
  mesh.write_cgal("test.mesh");
  Archiver arch = Archiver();
  arch.add_mat("basis",basis);
  arch.add_mat("ibasis",ibasis);

  arch.write("test.data");
}
