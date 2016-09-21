#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>
#include <armadillo>

#include "misc.h"
#include "io.h"
#include "tri_mesh.h"
#include "refine.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;
using namespace arma;
using namespace tri_mesh;

vec besselish_function(const Points & points,
                       const vec & center,
                       const double b){
  uint N = points.n_rows;
  assert(center.n_elem == points.n_cols);
  vec d = lp_norm(points - repmat(center.t(),N,1),2,1);
  return cos(b*b*d) % exp(-b*pow(d,2));
}

po::variables_map read_command_line(uint argc, char** argv){
  po::options_description desc("Meshing options");
  desc.add_options()
    ("help", "produce help message")
    ("infile_base,i", po::value<string>()->required(), "Input file base")
    ("outfile_base,o", po::value<string>()->required(),
     "Output (CGAL) mesh file base")
    ("mesh_angle", po::value<double>()->default_value(0.125),
     "Mesh angle refinement criterion")
    ("mesh_length", po::value<double>()->default_value(0.5),
     "Mesh edge length refinement criterion")    
    ("max_expansion", po::value<int>()->default_value(100),
     "Max number of cells to split")
    ("perc_expansion",po::value<double>()->default_value(0.1),
     "Percentage of cells to expand");
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

  string mesh_file = var_map["infile_base"].as<string>() + ".tri";

  // Read in the CGAL mesh
  TriMesh mesh;
  cout << "Reading in cgal mesh file [" << mesh_file << ']'  << endl;
  mesh.read_cgal(mesh_file);
  mesh.freeze();
  uint V = mesh.number_of_vertices();
  uint N = mesh.number_of_all_nodes();
  assert(N == V+1);
  uint F = mesh.number_of_faces();
  cout << "Mesh stats:"
       << "\n\tNumber of vertices: " << V
       << "\n\tNumber of faces: " << F
       << endl;
  Points centers = mesh.get_cell_centers();

  // Find boundary from the mesh and create the simulator object
  mat bbox = mesh.find_bounding_box();
  vec lb = bbox.col(0);
  vec ub = bbox.col(1);
  cout << "\tLower bound:" << lb.t()
       << "\tUpper bound:" << ub.t();

  // Read in solution information
  string soln_file = var_map["infile_base"].as<string>() + ".sol";
  cout << "Reading in LCP solution file [" << soln_file << ']'  << endl;
  Unarchiver sol_unarch(soln_file);
  vec p = sol_unarch.load_vec("p");

  string lcp_file = var_map["infile_base"].as<string>() + ".lcp";
  Unarchiver lcp_unarch(lcp_file);
  vec q = lcp_unarch.load_vec("q");

  Archiver arch;
  
  // Make sure that the primal information makes sense
  assert(0 == p.n_elem % V);
  uint A = p.n_elem / V;
  assert(A == 3);
  cout << "Blocking primal solution..."
       << "\n\tLength of primal solution: " << p.n_elem
       << "\n\tRatio of primal length to vertex number: " << A << endl;
  mat P = reshape(p,size(V,A));
  P = join_vert(P,datum::inf * ones<rowvec>(3)); // Pad
  vec value = P.col(0);
  mat flows = P.tail_cols(2);

  mat Q = reshape(q,size(V,A));
  Q = join_vert(Q,datum::inf * ones<rowvec>(3)); // Pad
  vec recon_b = mesh.interpolate(centers,
                                 conv_to<vec>::from(Q.col(1)));
  vec recon_c = mesh.interpolate(centers,
                                 conv_to<vec>::from(Q.col(2)));
  arch.add_vec("recon_b",recon_b);
  arch.add_vec("recon_c",recon_c);


  vec area = mesh.cell_area();
  arch.add_vec("area",area);

  // True values
  vec sq_dist = sum(pow(centers,2),1);
  vec b = sq_dist;
  vec c = max(zeros<vec>(F),1 - sq_dist);
  vec x = arma::min(b,c);
  assert(all(x >= 0));
  assert(F == x.n_elem);
  
  uvec pi = arma::index_min(join_horiz(b,c),1);
  assert(F == pi.n_elem);
  arch.add_uvec("pi",pi);

  // Approx policy
  assert(2 == flows.n_cols);
  mat interp_flows = mesh.interpolate(centers,flows);
  uvec flow_pi = arma::index_max(interp_flows,1);
  arch.add_uvec("flow_pi",flow_pi);

  assert(F == flow_pi.n_elem);
  uvec diff = zeros<uvec>(F);
  diff(find(flow_pi != pi)).fill(1);
  arch.add_uvec("policy_diff",diff);

  // Approx value
  vec interp_value = mesh.interpolate(centers,value);
  assert(F == interp_value.n_elem);
  vec res = abs(x - interp_value);
  arch.add_vec("residual",res);

  vec heuristic = res;
  heuristic(find(flow_pi != pi)) *= 4;
  arch.add_vec("heuristic",heuristic);

  double quant = 0.9;
  cout << "Quantile:" << quant << endl;
  double cutoff = quantile(heuristic,quant);
  cout << "\t Cutoff: " << cutoff
       << "\n\t Max:" << max(heuristic)
       << "\n\t Min:" << min(heuristic) << endl;

  // Split the cells if they have a large heuristic_1 or
  // policies disagree on them.
  TriMesh new_mesh(mesh);  
  Point center;
  uint new_nodes = 0;
  for(uint f = 0; f < F; f++){
    if(area(f) < 2e-4){
      continue;
    }
    if(new_nodes > 200) break;
    if(heuristic(f) > cutoff){
      center = convert(mesh.center_of_face(f));
      new_mesh.insert(center);
      new_nodes++;
    }
  }
  cout << "Added " << new_nodes << " new nodes..." << endl;
  
  cout << "Refining..." << endl;
  new_mesh.refine(0.125,1.0);
  new_mesh.lloyd(10);
  new_mesh.freeze();

  // Write out all the information
  string out_file_base = var_map["outfile_base"].as<string>();
  arch.write(out_file_base + ".stats");
  cout << "Writing..."
       << "\n\tCGAL mesh file: " << out_file_base << ".tri"
       << "\n\tShewchuk node file: " << out_file_base << ".node"
       << "\n\tShewchuk ele file: " << out_file_base << ".ele" << endl;
  new_mesh.write_cgal(out_file_base + ".tri");
  new_mesh.write_shewchuk(out_file_base);
}
