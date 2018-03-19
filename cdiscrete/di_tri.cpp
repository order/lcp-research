#include "di_tri.cpp"

void DoubleIntegratorSimulator::add_bang_bang_curve(TriMesh & mesh,
                                                    uint num_curve_points) const{
  VertexHandle v_zero = mesh.locate_vertex(Point(0,0));

  VertexHandle v_old = v_zero;
  VertexHandle v_new;  
  double x,y;
  double N = num_curve_points;

  vec lb = m_bbox.col(0);
  vec ub = m_bbox.col(1);
  
  // Figure out the max y within boundaries
  assert(lb(0) < 0);
  double max_y = min(ub(1),std::sqrt(-lb(0)));
  assert(max_y > 0);

  //Insert +ve y, -ve x points
  for(double i = 1; i < N; i++){
    y = max_y * i / N; // Uniform over y
    assert(y > 0);
    x = - y * y;
    if(x <= lb(0)) break;
    v_new = mesh.insert(Point(x,y));
    mesh.insert_constraint(v_old,v_new);
    v_old = v_new;
  }

  //Insert -ve y, +ve x points
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
}

TriMesh generate_initial_mesh(double angle, double length, const mat & bbox){
  TriMesh mesh;
  mesh.build_box_boundary(bbox);
  
  cout << "Refining based on (" << angle
       << "," << length <<  ") criterion ..."<< endl;
  mesh.refine(angle,length);
  
  cout << "Optimizing (25 rounds of Lloyd)..."<< endl;
  mesh.lloyd(25);
  
  cout << "Re-refining.."<< endl;
  mesh.refine(angle,length);

  mesh.freeze();
  return mesh;
}
