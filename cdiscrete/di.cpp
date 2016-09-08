#include "di.h"

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
