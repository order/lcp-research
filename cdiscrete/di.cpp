#include "di.h"


void add_di_bang_bang_curves(TriMesh & mesh,
			     VertexHandle & v_zero,
			     VertexHandle & v_upper_left,
			     VertexHandle & v_lower_right,
			     uint num_curve_points){
  VertexHandle v_old = v_zero;
  VertexHandle v_new;
  double x,y;
  double N = num_curve_points;
  // -ve x, +ve y
  for(double i = 1; i < N; i++){
    y = i / N; // Uniform over y
    x = - y * y;
    v_new = mesh.insert(Point(x,y));
    mesh.insert_constraint(v_old,v_new);
    v_old = v_new;
  }
  mesh.insert_constraint(v_old,v_upper_left);

  v_old = v_zero;
  for(double i = 1; i < N; i++){
    y = -i / N;
    x = y * y;
    v_new = mesh.insert(Point(x,y));
    mesh.insert_constraint(v_old,v_new);
    v_old = v_new;
  }
  mesh.insert_constraint(v_old,v_lower_right);
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
  
  mat costs = ones(N,2);
  costs.rows(l1_norm < 0.2).fill(0);  
  return costs;
}

vec build_di_state_weights(const Points & points){
  vec weight = sqrt(sum(pow(points,2),1));
  return weight / sum(weight);
}

int main()
{
  TriMesh mesh;
  
  VertexHandle v_low_left = mesh.insert(Point(-1,-1));
  VertexHandle v_low_right = mesh.insert(Point(1,-1));
  VertexHandle v_up_left = mesh.insert(Point(-1,1));
  VertexHandle v_up_right = mesh.insert(Point(1,1));
  VertexHandle v_zero = mesh.insert(Point(0,0));

  //Box boundary
  mesh.insert_constraint(v_low_left, v_low_right);
  mesh.insert_constraint(v_low_left, v_up_left);
  mesh.insert_constraint(v_up_right, v_low_right);
  mesh.insert_constraint(v_up_right, v_up_left);

  uint num_curve_points = 15;
  add_di_bang_bang_curves(mesh,v_zero,v_up_left,v_low_right,num_curve_points);
  
  mesh.refine(0.125,0.125);
  mesh.lloyd(25);
  mesh.freeze();
  mesh.write("test");

  cout << "Number of vertices: " << mesh.number_of_vertices() << endl;
  cout << "Number of faces: " << mesh.number_of_faces() << endl;

  // Generate interp grid
  vector<vec> grid;
  uint G = 150;
  grid.push_back(linspace(-1.0,1.0,G));
  grid.push_back(linspace(-1.0,1.0,G));
  Points grid_points = make_points(grid);
  ElementDist interp_weights = mesh.points_to_element_dist(grid_points);
  
  save_sp_mat(interp_weights,"test.grid");
 
  
  // Get points
  Points points = mesh.get_spatial_nodes();

  // Run dynamics
  Points p_pos = double_integrator(points,1,0.05);
  Points p_neg = double_integrator(points,-1,0.05);

  // Enforce boundary by saturation
  vec lb = -ones<vec>(2);
  vec ub = ones<vec>(2);
  saturate(p_pos,lb,ub);
  saturate(p_neg,lb,ub);

  // Convert to transition matrices
  ElementDist P_pos = mesh.points_to_element_dist(p_pos);
  ElementDist P_neg = mesh.points_to_element_dist(p_neg);

  // Crop out oob row (dealt with by saturation)
  uint N = points.n_rows;
  P_pos.resize(N,N);
  P_neg.resize(N,N);
  
  mat costs = build_di_costs(points);
  vec weights = build_di_state_weights(points);
  
  cout << "Discretizing..." << endl;

  //cout << distrib;

  save_sp_mat(P_pos,"p_pos.spmat");
  save_sp_mat(P_neg,"p_neg.spmat");
  save_mat(costs, "costs.mat");
  save_vec(weights, "weights.vec");
  
}
