#include "mesh.h"
#include <vector>

BaryCoord::BaryCoord():oob(true){}
BaryCoord::BaryCoord(bool o,uvec i,vec w) :
  oob(o),indices(i),weights(w){}
TriMesh::TriMesh() :
  m_mesh(),m_refiner(m_mesh),m_dirty(true),m_frozen(false){}

ostream& operator<< (ostream& os, const BaryCoord& coord){
  if(coord.oob){
    os << "OOB" << endl;
  }
  else{
    os << coord.indices.t() << coord.weights.t();
  }
  return os;
}

ElementDist TriMesh::points_to_element_dist(const Points & points){
  // Ignore CSC returns
  uvec row = uvec();
  uvec col = uvec();
  vec data = vec();
  return points_to_element_dist(points,row,col,data);
}

ElementDist TriMesh::points_to_element_dist(const Points & points,
					    uvec & row_idx_uvec,
					    uvec & col_ptr_uvec,
					    vec & data_vec){
  assert(2 == points.n_cols);
  assert(m_frozen);

  vector<uword> row_idx;
  vector<uword> col_ptr;
  vector<double> data;

  uint N = points.n_rows;
  uint oob_idx = oob_node_index();
  uint M = number_of_nodes();

  Point p;
  BaryCoord coord;

  // Set up sparse matrix via row indices and col pointers.
  // Assume elements are visited in col sorted order,
  // Column i is described by the row indices and data in the index range
  // col_ptr[i],...,col_ptr[i+1]-1 (empty if col_ptr[i] == col_ptr[i+1])

  uint oob_count;
  for(uint i = 0; i < N; i++){
    p = Point(points(i,0),points(i,1));
    coord = barycentric_coord(p);

    //cout << p << endl;
    
    // Start new column here
    col_ptr.push_back(row_idx.size());
    if(coord.oob){
      // Out of bounds; all weight on oob node
      assert(0 == coord.indices.n_elem);
      assert(0 == coord.weights.n_elem);     
      row_idx.push_back(oob_idx);
      data.push_back(1.0);
      oob_count++;
    }
    else{
      // In bounds; add barycentric coords.
      assert(NUMVERT == coord.indices.n_elem);
      assert(NUMVERT == coord.weights.n_elem);
      for(uint v = 0; v < NUMVERT; v++){
	if(coord.weights(v) < ALMOST_ZERO) continue;
	
	row_idx.push_back(coord.indices(v));
	data.push_back(coord.weights(v));
      }
    }
  }
  assert(row_idx.size() == data.size());
  assert(N == col_ptr.size());
  col_ptr.push_back(row_idx.size()); // Close off final column

  row_idx_uvec = uvec(row_idx);
  col_ptr_uvec = uvec(col_ptr);
  data_vec     = vec(data);
  
  return sp_mat(row_idx_uvec,col_ptr_uvec,data_vec,M,N);
}

BaryCoord TriMesh::barycentric_coord(const Point & point){  
  regen_caches();

  double x = point.x();
  double y = point.y();

  // Locate face
  int loc_int;
  LocateType loc;
  FaceHandle face = m_mesh.locate(point,
				  loc,loc_int);
  if(loc == CDT::OUTSIDE_CONVEX_HULL
     or loc == CDT::OUTSIDE_AFFINE_HULL){
    return BaryCoord(true,uvec(),vec()); // Out of bounds
  }
  
  // Extract vertices
  vec X = vec(3);
  vec Y = vec(3);
  uvec idx = uvec(3);
  for(uint i = 0; i < 3; i++){
    X(i) = face->vertex(i)->point().x();
    Y(i) = face->vertex(i)->point().y();
    idx(i) = m_vert_reg[face->vertex(i)];
  }

  // Barycentric voodoo (formula from wikipedia)
  vec c = vec(3);
  double Det = (Y(1) - Y(2))*(X(0) - X(2)) + (X(2) - X(1))*(Y(0) - Y(2));
  c(0) = ((Y(1) - Y(2))*(x - X(2)) + (X(2) - X(1))*(y - Y(2))) / Det;
  c(1) = ((Y(2) - Y(0))*(x - X(2)) + (X(0) - X(2))*(y - Y(2))) / Det;
  assert(0 <= c(0) and c(0) <= 1);
  assert(0 <= c(1) and c(1) <= 1);
  //cout << "\t" << (c(0) + c(1)) << endl;
  assert(c(0) + c(1) <= 1 + ALMOST_ZERO);
  c(2) = std::max(1.0 - c(0) - c(1),0.0);

  // Check the reconstruction
  vec p = vec({x,y});
  vec recon = vec(2);
  recon(0) = dot(X,c);
  recon(1) = dot(Y,c);
  assert(approx_equal(recon,p,"absdiff",ALMOST_ZERO));
  // TODO: return vertex indices too (need vertex registry)
  
  return BaryCoord(false,idx,c);
}

FaceHandle TriMesh::locate(const Point & p) const{
  return m_mesh.locate(p);
}

VertexHandle TriMesh::insert(vec & p){
  return insert(Point(p(0),p(1)));
}

VertexHandle TriMesh::insert(Point p){
  assert(not m_frozen);
  m_dirty = true;
  return m_mesh.insert(p);
}

void TriMesh::insert_constraint(VertexHandle & a, VertexHandle & b){
  assert(not m_frozen);
  m_dirty = true;
  m_mesh.insert_constraint(a,b);
}

void TriMesh::refine(double b, double s){
  assert(not m_frozen);
  m_dirty = true;
  m_refiner.set_criteria(MeshCriteria(b,s));
  m_refiner.refine_mesh();
}

void TriMesh::lloyd(uint I){
  assert(not m_frozen);
  m_dirty = true;
  CGAL::lloyd_optimize_mesh_2(m_mesh,
			      CGAL::parameters::max_iteration_number = I);
}

Points TriMesh::get_spatial_nodes(){
  assert(m_frozen);
  return m_nodes.head_rows(number_of_vertices());
}

Points TriMesh::get_all_nodes(){
  assert(m_frozen);
  return m_nodes;
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

uint TriMesh::number_of_nodes() const{
  // Number of spatial vertices + 1 oob node
  return number_of_vertices() + 1;
} 

uint TriMesh::oob_node_index() const{
  /*
    Only the one oob node so far
   */
  return m_mesh.number_of_vertices();
}

void TriMesh::freeze(){
  regen_caches();
  m_frozen = true;

}

void TriMesh::regen_caches(){
  if(!m_dirty) return;
  assert(not m_frozen);
  
  // Regenerate face and vertex caches
  m_nodes = mat(m_mesh.number_of_vertices()+1,2);
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
  m_nodes(v_id,0) = HUGE_VAL;
  m_nodes(v_id,1) = HUGE_VAL;

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

void save_mat(const mat & A,
	      string filename){
  uint N = A.n_rows;
  uint D = A.n_cols;
  uint header = 2;
  vec data = vec(header + N*D);
  data(0) = N;
  data(1) = D;
  data.tail(N*D) = vectorise(A);
  data.save(filename,raw_binary);  
}

void save_vec(const vec & v,
	      string filename){
  v.save(filename,raw_binary);  
}

void save_sp_mat(const sp_mat & A,
		 string filename){

  uint nnz = A.n_nonzero;
  uint N = A.n_rows;
  uint D = A.n_cols;

  uint header = 3;
  vec data = vec(header + 3*nnz);
  data(0) = N;
  data(1) = D;
  data(2) = nnz;

  typedef sp_mat::const_iterator SpIter;
  uint idx = 3;
  for(SpIter it = A.begin(); it != A.end(); ++it){
    data[idx++] = it.row();
    data[idx++] = it.col();
    data[idx++] = (*it);
  }
  assert(idx == data.n_elem);
  data.save(filename,raw_binary);
}

mat make_points(const vector<vec> & grids)
{
  // Makes a mesh out of the D vectors
  // 'C' style ordering... last column changes most rapidly
  
  // Figure out the dimensions of things
  uint D = grids.size();
  uint N = 1;
  for(vector<vec>::const_iterator it = grids.begin();
      it != grids.end(); ++it){
    N *= it->n_elem;
  }
  mat P = mat(N,D); // Create the matrix
  
  uint rep_elem = N; // Element repetition
  uint rep_cycle = 1; // Pattern rep
  for(uint d = 0; d < D; d++){
    uint n = grids[d].n_elem;
    rep_elem /= n;
    assert(N == rep_cycle * rep_elem * n);
    
    uint I = 0;
    for(uint c = 0; c < rep_cycle; c++){ // Cycle repeat
      for(uint i = 0; i < n; i++){ // Element in pattern
	for(uint e = 0; e < rep_elem; e++){ // Element repeat
	  assert(I < N);
	  P(I,d) = grids[d](i);
	  I++;
	}
      }
    }
    rep_cycle *= n;
  }
  return P;
}
