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

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;
using namespace arma;
using namespace tri_mesh;

void generate_minop_mesh(TriMesh & mesh,
                         const string & filename){
  double angle = 0.125;
  double length = 0.075;

  cout << "Initial meshing..."<< endl;
  mesh.build_box_boundary({{-1.1,1.1},{-1.1,1.1}});
  mesh.build_circle(zeros<vec>(2),50,1.0);
  mesh.build_circle(zeros<vec>(2),30,1.0/sqrt(2.0));
  mesh.build_circle(zeros<vec>(2),25,0.25);


  cout << "Refining based on (" << angle
       << "," << length <<  ") criterion ..."<< endl;
  mesh.refine(angle,length);
  
  cout << "Optimizing (25 rounds of Lloyd)..."<< endl;
  mesh.lloyd(25);
  mesh.refine(angle,length);
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

po::variables_map read_command_line(uint argc, char** argv){
  po::options_description desc("Meshing options");
  desc.add_options()
    ("help", "produce help message")
    ("outfile_base,o", po::value<string>()->required(),"Output plcp file base");
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

mat make_ring_basis(const Points & points,
               uint S){
  
  vec dist = lp_norm(points,2,1);

  uint V = points.n_rows;
  mat basis = zeros<mat>(V,S);
  double l,u;
  double inc_rad = 1.0 / (double) S;
  uvec mask;
  for(uint s = 0; s < S; s++){
    l = inc_rad * s;
    u = inc_rad * (s + 1);
    if(s == S-1) u += 1e-12;
    mask = find(dist >= l and dist < u);
    if(0 == mask.n_elem)
      basis.col(s).randu();
    else
      basis(mask, uvec{s}).fill(1);
  }
  basis = orth(basis);
  return basis;
}

vec make_spike_basis(const Points & points,
                const vec & center){
  uint V = points.n_rows;
  vec dist = lp_norm(points.each_row() - center.t(),2,1);
  double upper_rad = max(dist);
  double lower_rad = 0;
  double rad;
  uint count;
  cout << "Finding spike radius: " << endl;
  while(true){
    rad = (upper_rad + lower_rad) / 2.0;
    count = (conv_to<uvec>::from(find(dist <= rad))).n_elem;
    cout << "\t At r=" << rad << ", "
         << count << " nodes included" << endl;
    assert(count <= V);

    if(count == 3){
      break;
    }
    if(count < 3){
      lower_rad = rad;
    }
    else{
      upper_rad = rad;
    }
  }
  
  vec basis = zeros<vec>(V);
  basis(find(dist <= rad)).fill(1);
  basis = orth(basis,2);
  return basis;
}

mat make_grid_basis(const Points & points,
               const mat & bbox,
               uint X, uint Y){
  uint V = points.n_rows;
  double dx = (bbox(0,1) - bbox(0,0)) / (double)X;
  double dy = (bbox(1,1) - bbox(1,0)) / (double)Y;
  mat basis = zeros<mat>(V,X*Y);
  uint b = 0;
  uvec mask;
  double xl,yl,xh,yh;
  for(uint i = 0; i < X;i++){
    assert(b < X*Y);
    xl = (double)i*dx + bbox(0,0);
    xh = xl + dx;
    if(i == X-1) xh += 1e-12;
    for(uint j = 0; j < Y; j++){
      yl = (double)j*dy + bbox(1,0);
      yh = yl + dy;
      if(j == Y-1) yh += 1e-12;
      mask = find(points.col(0) >= xl
                  and points.col(0) < xh
                  and points.col(1) >= yl
                  and points.col(1) < yh);
      basis(mask,uvec{b}).fill(1);
      b++;
    }
  }
  basis = orth(basis);
  return basis;
}

LCP build_minop_lcp(const TriMesh &mesh,
                    const vec & a,
                    const vec & b,
                    const vec & c){
  uint N = mesh.number_of_spatial_nodes();
  Points points = mesh.get_spatial_nodes();
  
  assert(a.n_elem == b.n_elem);
  assert(all(a < 0));
  vec q = join_vert(a,
                    join_vert(b,c));
  assert(3*N == q.n_elem);
  assert(not all(q >= 0));
 
  vector<sp_mat> E;
  E.push_back(speye(N,N));
  E.push_back(speye(N,N));
  sp_mat M = build_M(E);
  assert(M.is_square());
  assert(3*N == M.n_rows);

  return LCP(M,q);
}

int main(int argc, char** argv)
{
  po::variables_map var_map = read_command_line(argc,argv);
  string filename = var_map["outfile_base"].as<string>();
  
  // Read in the CGAL mesh
  TriMesh mesh;
  generate_minop_mesh(mesh,filename);
  mesh.freeze();
  
  uint V = mesh.number_of_vertices();
  uint F = mesh.number_of_faces();
  cout << "Mesh stats:"
       << "\n\tNumber of vertices: " << V
       << "\n\tNumber of faces: " << F
       << endl;

  // Find boundary from the mesh and create the simulator object
  mat bbox = mesh.find_bounding_box();

  double off = 1;
  Points points = mesh.get_spatial_nodes();
  vec sq_dist = sum(pow(points,2),1);
  vec a = -ones<vec>(V) / (double) V;
  vec b = sq_dist + off;
  vec c = max(zeros<vec>(V),1 - sq_dist) + off;
  
  LCP lcp = build_minop_lcp(mesh,a,b,c);
  string lcp_file = filename + ".lcp";
  cout << "Writing to " << lcp_file << endl;
  lcp.write(lcp_file);
  
  mat ring_basis = make_ring_basis(points,10);
  mat grid_basis = make_grid_basis(points,bbox,10,10);  

  //ring_basis = join_horiz(ring_basis,b);
  //ring_basis = join_horiz(ring_basis,c);
  ring_basis = orth(join_horiz(ones<vec>(V),ring_basis));

  //grid_basis = join_horiz(grid_basis,ones<vec>(V));
  grid_basis = orth(grid_basis);
  
  block_sp_row D;
  //D.push_back(sp_mat(grid_basis));
  //D.push_back(sp_mat(ring_basis));
  //D.push_back(speye(V,V));
  
  D.push_back(sp_mat(ring_basis));
  D.push_back(sp_mat(grid_basis));
  D.push_back(sp_mat(grid_basis));
  sp_mat Phi = diags(D);
  sp_mat U = Phi.t() * (lcp.M + 1e-9 * speye(3*V,3*V));// * Phi * Phi.t();
  vec r =  Phi.t() * lcp.q;

  string plcp_file = filename + ".plcp";
  cout << "Writing to " << plcp_file << endl;
  Archiver plcp_arch;
  plcp_arch.add_sp_mat("Phi",Phi);
  plcp_arch.add_sp_mat("U",U);
  plcp_arch.add_vec("r",r);
  plcp_arch.add_vec("a",a);
  plcp_arch.add_vec("b",b);
  plcp_arch.add_vec("c",c);
  plcp_arch.write(plcp_file);
}
