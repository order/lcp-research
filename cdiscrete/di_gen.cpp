#include <boost/program_options.hpp>

#include "di_gen.h"
#include "misc.h"
#include "io.h"

namespace po = boost::program_options;

void add_di_bang_bang_curves(TriMesh & mesh,
			     const vec & lb,
			     const vec & ub,
			     uint num_curve_points){
  VertexHandle v_zero = mesh.locate_vertex(Point(0,0));
  VertexHandle v_01 = mesh.locate_vertex(Point(lb(0),ub(1)));
  VertexHandle v_10 = mesh.locate_vertex(Point(ub(0),lb(1)));

  VertexHandle v_old = v_zero;
  VertexHandle v_new;  
  double x,y;
  double N = num_curve_points;
  // -ve x, +ve y

  // Figure out the max y within boundaries
  assert(lb(0) < 0);
  double max_y = min(ub(1),std::sqrt(-lb(0)));
  assert(max_y > 0);
  for(double i = 1; i < N; i++){
    y = max_y * i / N; // Uniform over y
    assert(y > 0);
    x = - y * y;
    if(x <= lb(0)) break;
    v_new = mesh.insert(Point(x,y));
    mesh.insert_constraint(v_old,v_new);
    v_old = v_new;
  }
  //mesh.insert_constraint(v_old,v_01);

  v_old = v_zero;
  assert(ub(0) > 0);
  double min_y = max(lb(1),-std::sqrt(ub(0)));
  assert(min_y < 0);
  for(double i = 1; i < N; i++){
    y = min_y * i / N;
    assert(y < 0);
    x = y * y;
    if(x >= ub(0)) break;
    v_new = mesh.insert(Point(x,y));
    mesh.insert_constraint(v_old,v_new);
    v_old = v_new;
  }
  //mesh.insert_constraint(v_old,v_10);
}

void saturate(Points & points,
	      const vec &lb,
	      const vec &ub){
  uint D = points.n_cols;
  assert(D == lb.n_elem);
  assert(D == ub.n_elem);

  uvec mask;
  uvec col_idx;
  for(uint d = 0; d < D; d++){
    col_idx = uvec({d});
    mask = find(points.col(d) > ub(d));
    points(mask,col_idx).fill(ub(d));

    mask = find(points.col(d) < lb(d));
    points(mask,col_idx).fill(lb(d));    
  }
}

Points double_integrator(const Points & points,
			 double a,double t){
  assert(2 == points.n_cols);
  Points new_points = Points(size(points));
  assert(size(points) == size(new_points));
  
  new_points.col(0) = points.col(0) + t * points.col(1) + 0.5 * t*t * a;
  new_points.col(1) = points.col(1) + t*a;
  
  return new_points;
}

mat build_di_costs(const Points & points){
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(2 == D);

  vec l1_norm = sum(abs(points),1);
  assert(N == l1_norm.n_elem);
  assert(all(l1_norm >= 0));
  
  mat costs = ones(N,2);
  costs.rows(find(l1_norm < 0.2)).fill(0);
  return costs;
}

vec build_di_state_weights(const Points & points){
  vec weight = sqrt(sum(pow(points,2),1));
  return weight / sum(weight);
}

sp_mat build_di_transition(const Points & points,
			   const TriMesh & mesh,
			   const vec & lb,
			   const vec & ub,
			   double action){
  uint N = points.n_rows;
  Points p_next = double_integrator(points,action,SIM_STEP);
  saturate(p_next,lb,ub);
  ElementDist ret = mesh.points_to_element_dist(p_next);

  
  // Final row is the OOB row
  assert(size(N+1,N) == size(ret));
  // Should be all zero
  assert(arma::accu(ret.submat(span(N,N),span(0,N-1))) < 1e-15);

  // Crop
  ret.resize(N,N);
  return ret;
}

void build_square_boundary(TriMesh & mesh,
			   const vec & lb,
			   const vec & ub){
  assert(2 == lb.n_elem);
  assert(2 == ub.n_elem);

  VertexHandle v_00 = mesh.insert(Point(lb(0),lb(1)));
  VertexHandle v_01 = mesh.insert(Point(lb(0),ub(1)));
  VertexHandle v_10 = mesh.insert(Point(ub(0),lb(1)));
  VertexHandle v_11 = mesh.insert(Point(ub(0),ub(1)));

  mesh.insert_constraint(v_00,v_01);
  mesh.insert_constraint(v_01,v_11);
  mesh.insert_constraint(v_11,v_10);
  mesh.insert_constraint(v_10,v_00);
}

bool check(const sp_mat & A){
  typedef sp_mat::const_iterator sp_it;
  set<pair<uint,uint>> S;
  for(sp_it it = A.begin();
      it != A.end(); ++it){
    pair<uint,uint> coord = make_pair(it.row(),it.col());
    if(S.end() != S.find(coord)){
      cout << "Violation: (" << it.row() << "," << it.col() << ")\n";
      assert(S.end() == S.find(coord));
    }
    S.emplace(coord);
  }
  assert(S.size() == A.n_nonzero);
}

void generate_initial_mesh(const po::variables_map & var_map,
                           TriMesh & mesh){
  double B = var_map["boundary"].as<double>();    
  uint num_bang_points = var_map["bang_points"].as<uint>();
  double angle = var_map["mesh_angle"].as<double>();
  double length = var_map["mesh_length"].as<double>();
  
  vec lb = -B*ones<vec>(2);
  vec ub = B*ones<vec>(2);

  cout << "Initial meshing..."<< endl;  
  build_square_boundary(mesh,lb,ub);
  VertexHandle v_zero = mesh.insert(Point(0,0));

  if(num_bang_points > 0){
    add_di_bang_bang_curves(mesh,lb,ub,num_bang_points);
  }
  
  cout << "Refining based on (" << angle << "," << length <<  ") criterion ..."<< endl;
  mesh.refine(angle,length);
  
  cout << "Optimizing (10 rounds of Lloyd)..."<< endl;
  mesh.lloyd(10);
  mesh.refine(angle,length);

  mesh.freeze();

  // Write initial mesh to file
  string filename =  var_map["mesh_length"].as<string>();
  cout << "Writing:"
       << "\n\t" << (filename + "node") << " (Shewchuk node file)"
       << "\n\t" << (filename + ".ele") << " (Shewchuk element file)"
       << "\n\t" << (filename + ".tri") << " (CGAL mesh file)" << endl;
  mesh.write_shewchuk("di");
  mesh.write_cgal("di.tri");
}

void build_lcp(const po::variables_map & var_map,
               const TriMesh & mesh){
  mat bounds = mesh.find_box_boundary();
  vec lb = bounds.col(0);
  vec ub = bounds.col(1);
  
  Points points = mesh.get_spatial_nodes();
  uint N = points.n_rows;
  
  cout << "Generating transition matrices..."<< endl;
  // Get transition matrices
  ElementDist P_pos = build_di_transition(points,
					  mesh,
					  lb,ub,1.0);
  ElementDist P_neg = build_di_transition(points,
					  mesh,
					  lb,ub,-1.0);

  double gamma = var_map["gamma"].as<double>();
  sp_mat E_pos = speye(N,N) + ( -gamma * P_pos);
  sp_mat E_neg = speye(N,N) - gamma * P_neg;
  
  sp_mat E_pos_t = E_pos.t();
  cout << "Building LCP..."<< endl;
  block_sp_row top = block_sp_row{sp_mat(),E_pos,E_neg};
  block_sp_row middle = block_sp_row{-E_pos.t(),sp_mat(),sp_mat()};
  block_sp_row bottom = block_sp_row{-E_neg.t(),sp_mat(),sp_mat()};
  block_sp_mat blk_M = block_sp_mat{top,middle,bottom};
  sp_mat M = bmat(blk_M);
  
  mat costs = build_di_costs(points);
  vec weights = build_di_state_weights(points);

  vec q = join_vert(-weights,vectorise(costs));
  assert(3*N == q.n_elem);

  string filename = var_map["base_file"].as<string>() + ".lcp";
  cout << "Writing " << filename << endl;
  Archiver archiver;
  archiver.add_sp_mat("M",M);
  archiver.add_vec("q",q);
  archiver.write(filename);
}


po::variables_map read_command_line(uint argc, char** argv){
  po::options_description desc("Meshing options");
  desc.add_options()
    ("help", "produce help message")
    ("base_file", po::value<string>()->required(),
     "Prefix for all files generated")
    ("mesh_file", po::value<string>(), "Input mesh file")
    ("mesh_angle", po::value<double>()->default_value(0.125),
     "Mesh angle refinement criterion")
    ("mesh_length", po::value<double>()->default_value(0.5),
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

//===========================================================
// Main function

int main(int argc, char** argv)
{
  po::variables_map var_map = read_command_line(argc,argv);

  TriMesh mesh;
  if(var_map.count("mesh_file")){
    string mesh_file = var_map["mesh_file"].as<string>();
    cout << "Reading in mesh from [" << mesh_file << "]..." << endl;
    mesh.read_cgal(mesh_file);
  }
  else{
    cout << "Generating initial mesh..." << endl;
    generate_initial_mesh(var_map,mesh);
  }

  mat bounds = mesh.find_box_boundary();
  vec lb = bounds.col(0);
  vec ub = bounds.col(1);
  
  cout << "Mesh stats:" << endl;
  cout << "\tNumber of vertices: " << mesh.number_of_vertices() << endl;
  cout << "\tNumber of faces: " << mesh.number_of_faces() << endl;
  cout << "\tLower bound:" << lb.t();
  cout << "\tUpper bound:" << ub.t();

  return 1;
  // Get points

}
