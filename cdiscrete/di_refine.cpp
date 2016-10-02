#include "di.h"
#include "misc.h"
#include "io.h"
#include "lcp.h"
#include "solver.h"
#include "refine.h"

#define CGAL_MESH_2_OPTIMIZER_VERBOSE

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace tri_mesh;

po::variables_map read_command_line(uint argc, char** argv){
  po::options_description desc("Meshing options");
  desc.add_options()
    ("help", "produce help message")
    ("outfile_base,o", po::value<string>()->required(),
     "Base for all files generated (lcp, mesh)")
    ("mesh_file,m", po::value<string>(), "Input (CGAL) mesh file")
    ("mesh_angle", po::value<double>()->default_value(0.11),
     "Mesh angle refinement criterion")
    ("mesh_length", po::value<double>()->default_value(1),
     "Mesh edge length refinement criterion")    
    ("gamma,g", po::value<double>()->default_value(0.997),
     "Discount factor")
    ("boundary,B", po::value<double>()->default_value(5.0),
     "Square boundary box [-B,B]^2")
    ("bang_points,b", po::value<uint>()->default_value(0),
     "Number of bang-bang curve points to add to initial mesh");
  po::variables_map var_map;
  po::store(po::parse_command_line(argc, argv, desc), var_map);
  po::notify(var_map);

  if (var_map.count("help")) {
    cout << desc << "\n";
    exit(1);
  }
  return var_map;
}

DoubleIntegratorSimulator build_di_simulator(const po::variables_map & var_map){
  double B = var_map["boundary"].as<double>();
  mat bbox = mat(2,2);
  bbox.col(0).fill(-B);
  bbox.col(1).fill(B);
  mat actions = vec{-1,1};
  return DoubleIntegratorSimulator(bbox,actions);
}

void generate_initial_mesh(const po::variables_map & var_map,
                           const DoubleIntegratorSimulator & di,
                           TriMesh & mesh){
  uint num_bang_points = var_map["bang_points"].as<uint>();
  double angle = var_map["mesh_angle"].as<double>();
  double length = var_map["mesh_length"].as<double>();

  double B = var_map["boundary"].as<double>();    
  vec lb = -B*ones<vec>(2);
  vec ub = B*ones<vec>(2);
  mat bbox = join_horiz(lb,ub);

  cout << "Initial meshing..."<< endl;
  mesh.build_box_boundary(lb,ub);
  VertexHandle v_zero = mesh.insert(Point(0,0));

  if(num_bang_points > 0){
    di.add_bang_bang_curve(mesh,num_bang_points);
  }
  
  cout << "Refining based on (" << angle
       << "," << length <<  ") criterion ..."<< endl;
  mesh.refine(angle,length);
  
  cout << "Optimizing (10 rounds of Lloyd)..."<< endl;
  mesh.lloyd(10);
  mesh.refine(angle,length);

  mesh.freeze();
}

umat generate_policy_mat(const TriMesh & mesh,
                         const DoubleIntegratorSimulator & di,
                         const vec & value,
                         const mat & flows,
                         double gamma){
  uint F = mesh.number_of_faces();
  
  uvec f_pi = flow_policy(&mesh,flows);  
  uvec g_pi = grad_policy(&mesh,&di,value);
  uvec q_pi = q_policy(&mesh,&di,value,gamma);

  umat policies = umat(F,3);
  policies.col(0) = f_pi;
  policies.col(1) = g_pi;
  policies.col(2) = q_pi;
  return policies;
}

bvec find_policy_disagreement(const umat & policies){
  uint N = policies.n_rows;
  uint C = policies.n_cols;
  uvec agg = sum(policies,1);
  assert(N == agg.n_elem);
  bvec disagree = ones<bvec>(N);
  disagree(find(0 == agg)).fill(0);
  disagree(find(C == agg)).fill(0);
  return disagree;
}

vec residual_heuristic(const TriMesh & mesh,
                                const DoubleIntegratorSimulator & di,
                                const vec & value,
                                const vec & flow_vol,
                                double gamma){
  uint N = mesh.number_of_spatial_nodes();
  assert(N == value.n_elem);
  
  uint F = mesh.number_of_faces();
  assert(F == flow_vol.n_elem);
  
  vec bell_res = bellman_residual(&mesh,&di,value,gamma,0,25);
  assert(F == bell_res.n_elem);

  vec H = bell_res % pow(flow_vol,0.5);
  assert(F == H.n_elem);
  return H;
}

vec policy_disagreement_heuristic(const TriMesh & mesh,
                                           const DoubleIntegratorSimulator & di,
                                           const vec & value,
                                           const vec & flow_vol,
                                           const bvec & disagree,
                                           double gamma){
  uint F = mesh.number_of_faces();
  vec adv_fun = advantage_function(&mesh,&di,value,gamma,0,25);
  assert(F == adv_fun.n_elem);
  vec H = adv_fun % flow_vol;
  H(find(0 == disagree)).fill(0);
  assert(F == H.n_elem);
  return H;
}

uint refine_mesh(const po::variables_map & var_map,
                 TriMesh & mesh,
                 const vec & res_heur,
                 const vec & pol_heur){
  double angle = var_map["mesh_angle"].as<double>();
  double length = var_map["mesh_length"].as<double>();

  uint N = mesh.number_of_spatial_nodes();
  uint F = mesh.number_of_faces();
  assert(F == res_heur.n_rows);
  assert(F == pol_heur.n_rows);
  
  double res_cutoff = quantile(res_heur,0.95);
  double pol_cutoff = quantile(pol_heur,0.95);
  vec area = mesh.cell_area();

  // Identify the centers we want to expand
  vector<Point> new_points;
  for(uint f = 0; f < F; f++){
    if(area(f) < 0.01)
      continue; // Too small
    
    if(res_heur(f) > res_cutoff
       or pol_heur(f) > pol_cutoff){
      Point new_point = convert(mesh.center_of_face(f));
      new_points.push_back(new_point);
    }
  }
  mesh.unfreeze();
  
  // Add these centers (two steps to avoid indexing issues)
  cout << "\tAdding " << new_points.size() << " new points..." << endl;
  for(vector<Point>::const_iterator it = new_points.begin();
      it != new_points.end();++it){
    mesh.insert(*it);
  }

  cout << "\tOptimizing node position..." << endl;
  mesh.lloyd(250);
  cout << "\tRefining split cells..." << endl;
  mesh.refine(angle,length);
  mesh.freeze();

  return mesh.number_of_spatial_nodes() - N;
}

//===========================================================
// Main function

int main(int argc, char** argv)
{
  // Parse command line
  po::variables_map var_map = read_command_line(argc,argv);

  cout << "Generating initial mesh..." << endl;
  TriMesh mesh;
  DoubleIntegratorSimulator di = build_di_simulator(var_map);
  generate_initial_mesh(var_map,di,mesh);
  mesh.freeze();

  double gamma = var_map["gamma"].as<double>();

  cout << "Initializing solver..." << endl;
  KojimaSolver solver;
  solver.comp_thresh = 1e-8;
  solver.max_iter = 500;
  solver.regularizer = 1e-8;
  solver.verbose = true;
  solver.aug_rel_scale = 10;

  string filebase = var_map["outfile_base"].as<string>();
  uint A = 2;
  for(uint i = 0; i < 1; i++){
    string iter_filename = filebase + "." + to_string(i);
    
    uint N = mesh.number_of_spatial_nodes();
    uint F = mesh.number_of_faces();

    // These are the mesh files for this iteration
    cout << "Writing:"
         << "\n\t" << (iter_filename + ".node") << " (Shewchuk node file)"
         << "\n\t" << (iter_filename + ".ele") << " (Shewchuk element file)"
         << "\n\t" << (iter_filename + ".tri") << " (CGAL mesh file)" << endl;
    mesh.write_shewchuk(iter_filename);
    mesh.write_cgal(iter_filename + ".tri");

    cout << "Building LCP..." << endl;
    bool include_oob = false;
    bool value_nonneg = true;
    LCP lcp = build_lcp(&di,&mesh,gamma,include_oob,value_nonneg);
    
    cout << "Solving iteration " << i << " LCP..." << endl;
    SolverResult sol = solver.aug_solve(lcp);
    assert((A+1)*N == sol.p.n_elem);
    sol.write(iter_filename + ".sol");
    cout << "Sol len: " << sol.p.n_elem << endl;
    
    mat P = reshape(sol.p,size(N,(A+1)));
    vec value = P.col(0);
    mat flows = P.tail_cols(2);

    cout << "Calculating heuristics..." << endl;
    umat policies =  generate_policy_mat(mesh, di, value, flows, gamma);
    bvec disagree = find_policy_disagreement(policies);
    vec flow_vol = mesh.prism_volume(sum(flows,1));
    vec res_heur = residual_heuristic(mesh,di,value,flow_vol,gamma);
    vec pol_heur = policy_disagreement_heuristic(mesh,di,value,
                                                 flow_vol,
                                                 disagree,gamma);


    cout << "Refining mesh..." << endl;
    uint num_new_nodes = refine_mesh(var_map,
                                     mesh,
                                     res_heur,
                                     pol_heur);
    cout << "Added " << num_new_nodes << " new nodes.";
  }
}
