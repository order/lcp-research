#include "mesh.h"

TriMesh::TriMesh():
  m_mesh(),
  m_refiner(m_mesh){}

vec TriMesh::barycentric_coord(const vec & p) const{
  assert(2 == point.n_elem);

  double x = p(0);
  double y = p(1);
  
  // Locate face
  Point cgal_point = Point(x,y);
  FaceHandle face = m_mesh.locate(cgal_point);

  // Extract vertices
  vec X = vec(3);
  vec Y = vec(3);
  for(uint i = 0; i < 3; i++){
    X(i) = face->vertex(i)->point().x();
    Y(i) = face->vertex(i)->point().y();
  }

  // Barycentric voodoo (formula from wikipedia)
  vec c = vec(3);
  double Det = (Y(1) - Y(2))*(X(0) - X(2)) + (X(2) - X(1))*(Y(0) - Y(2));
  c(0) = ((Y(1) - Y(2))*(x - X(2)) + (X(2) - X(1))*(y - Y(2))) / Det;
  c(1) = ((Y(2) - Y(0))*(x - X(2)) + (X(0) - X(2))*(y - Y(2))) / Det;
  c(2) = 1.0 - c(0) - c(1);

  // Check the reconstruction 
  vec recon = vec(2);
  recon(0) = dot(X,c);
  recon(1) = dot(Y,c);
  assert(approx_equal(recon,p,"reldiff",1e-12));

  // TODO: return vertex indices too (need vertex registry)
  return c;
}

VertexHandle TriMesh::insert(vec & p){
  return insert(Point(p(0),p(1)));
}

VertexHandle TriMesh::insert(Point p){
  m_dirty = true;
  return m_mesh.insert(p);
}

void TriMesh::insert_constraint(VertexHandle & a, VertexHandle & b){
  m_dirty = true;
  m_mesh.insert_constraint(a,b);
}

void TriMesh::refine(double b, double s){
  m_dirty = true;
  m_refiner.set_criteria(MeshCriteria(b,s));
  m_refiner.refine_mesh();
}

void TriMesh::lloyd(uint I){
  m_dirty = true;
  CGAL::lloyd_optimize_mesh_2(m_mesh,
			      CGAL::parameters::max_iteration_number = I);
}

void TriMesh::write(string base_filename){
  // Write the .node and .ele files. Shewchuk uses these files in Triangle
  // and Stellar
  ofstream node_file,ele_file;
  uint attr = 0; // Number of attributes, will be useful later
  uint bnd = 0; // Boundary marker. Will be important for

  // Regenerate all the supporting information
  regen_caches();
  
  // Write .node header
  string node_filename = base_filename + ".node";
  node_file.open(node_filename.c_str());
  
  // <# of vertices> <dim> <# of attributes> <# of boundary markers (0 or 1)>
  node_file << m_nodes.n_rows
	    << "\t" << NUMDIM
	    << "\t" << attr
	    << "\t" << bnd << endl;
  
  // <vertex #> <x> <y> [attributes] [boundary marker] 
  for(uint i = 0; i < m_nodes.n_rows; i++){
    node_file << i << "\t" << m_nodes.row(i);
  }
  node_file.close();

  // Write .ele file
  string ele_filename = base_filename + ".ele";
  ele_file.open(ele_filename.c_str());
    // <# of triangles> <nodes per triangle> <# of attributes>
  ele_file << m_faces.n_rows
	   << "\t" << NUMVERT
	   << "\t" << attr << endl;
  
  for(uint i = 0; i < m_faces.n_rows; i++){
    ele_file << i << "\t" << m_faces.row(i);
  }
  ele_file.close();
}

uint TriMesh::number_of_faces() const{
  return m_mesh.number_of_faces();
}
uint TriMesh::number_of_vertices() const{
  return m_mesh.number_of_vertices();
}

void TriMesh::regen_caches(){
  if(!m_dirty) return;
  
  // Regenerate face and vertex caches
  m_nodes = mat(m_mesh.number_of_vertices(),2);
  m_faces = umat(m_mesh.number_of_faces(),3);
  m_vert_reg.clear();
  m_face_reg.clear();

  uint v_id = 0;
  for(VertexIterator vit = m_mesh.vertices_begin();
      vit != m_mesh.vertices_end(); ++vit){
    // Add to point mat
    m_nodes(v_id,0) = vit->point().x();
    m_nodes(v_id,1) = vit->point().y();
    // Register
    m_vert_reg[vit] = v_id++;
  }

  uint f_id = 0;
  for(FaceIterator fit = m_mesh.faces_begin();
      fit != m_mesh.faces_end(); ++fit){
    // Add to face description
    for(uint v = 0; v < NUMVERT; v++){
      m_faces(f_id,v) = m_vert_reg[fit->vertex(v)];
    }
    m_face_reg[fit] = f_id++;
  }
  m_dirty = false;
}

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
  
  mesh.refine(0.125,0.2);
  mesh.lloyd(25);

  cout << "Number of vertices: " << mesh.number_of_vertices() << endl;
  cout << "Number of faces: " << mesh.number_of_faces() << endl;


  mesh.write("test");
}
