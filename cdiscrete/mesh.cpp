#include "mesh.h"
#include "misc.h"
#include <vector>


Point convert(const vec & point){
  assert(2 == point.n_elem);
  return Point(point(0),point(1));
}

BaryCoord::BaryCoord():oob(true){}
BaryCoord::BaryCoord(bool o,uvec i,vec w) :
  oob(o),indices(i),weights(w){}

TriMesh::TriMesh() :
  m_mesh(),m_refiner(m_mesh),m_dirty(true),m_frozen(false){}

TriMesh::TriMesh(const TriMesh & other) :
  m_mesh(other.m_mesh),m_refiner(m_mesh),m_dirty(true),m_frozen(false){
  regen_caches();
}

ostream& operator<< (ostream& os, const BaryCoord& coord){
  if(coord.oob){
    os << "OOB" << endl;
  }
  else{
    os << coord.indices.t() << coord.weights.t();
  }
  return os;
}

ElementDist TriMesh::points_to_element_dist(const Points & points) const{
  // Ignore CSC returns
  uvec row = uvec();
  uvec col = uvec();
  vec data = vec();
  return points_to_element_dist(points,row,col,data);
}

ElementDist TriMesh::points_to_element_dist(const Points & points,
					    uvec & row_idx_uvec,
					    uvec & col_ptr_uvec,
					    vec & data_vec) const{
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
      assert(TRI_NUM_VERT == coord.indices.n_elem);
      assert(TRI_NUM_VERT == coord.weights.n_elem);
      
      for(uint v = 0; v < TRI_NUM_VERT; v++){
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

template <typename T> T TriMesh::interpolate(const Points & points,
                                    const T& data) const{
  assert(m_frozen);
  
  uint N = points.n_rows;
  uint d = points.n_cols;
  uint NN = number_of_nodes();
  assert(data.n_rows == NN);
  
  ElementDist D = points_to_element_dist(points);
  assert(size(D) == size(NN,N));

  T ret = D.t() * data;
  assert(ret.n_rows == N);
  assert(ret.n_cols == data.n_cols);
  return ret;
}
template mat TriMesh::interpolate<mat>(const Points &, const mat&) const;
template vec TriMesh::interpolate<vec>(const Points &, const vec&) const;


BaryCoord TriMesh::barycentric_coord(const Point & point) const{  
  assert(m_frozen);

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
  
  // Extract vertices; sort by id
  /*
    ID sorting turns out to be important for turning into a sparse matrix later.
   */
  vector< tuple<uint, double, double> > vertex_list;
  set<uint> ids;
  uint id;
  double v_x,v_y;
  for(uint i = 0; i < 3; i++){
    id = m_vert_reg.at(face->vertex(i));
    v_x = face->vertex(i)->point().x();
    v_y = face->vertex(i)->point().y();
    assert(ids.end() == ids.find(id)); // Unique
    vertex_list.push_back(make_tuple(id,v_x,v_y));
  }
  sort(vertex_list.begin(),vertex_list.end());
  
  vec X = vec(3);
  vec Y = vec(3);
  uvec idx = uvec(3);
  for(uint i = 0; i < 3; i++){
    idx(i) = get<0>(vertex_list[i]);
    X(i) = get<1>(vertex_list[i]);
    Y(i) = get<2>(vertex_list[i]);
  }
  
  // Barycentric voodoo (formula from wikipedia)
  // TODO: could have replaced this with stuff from Triangle_coordinates
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
  if(accu(abs(recon-p)) > PRETTY_SMALL){
    cout << "Abs reconstruction error: " << accu(abs(recon-p)) << endl;
    assert(approx_equal(recon,p,"absdiff",PRETTY_SMALL));
  }
  // TODO: return vertex indices too (need vertex registry)
  
  return BaryCoord(false,idx,c);
}

FaceHandle TriMesh::locate_face(const Point & p) const{
  LocateType lt;
  int li;
  FaceHandle ret = m_mesh.locate(p,lt,li);
  assert(lt != CDT::OUTSIDE_CONVEX_HULL);
  assert(lt != CDT::OUTSIDE_AFFINE_HULL);
  return ret;
}
VertexHandle TriMesh::locate_vertex(const Point & p) const{
  LocateType lt;
  int li;
  FaceHandle face = m_mesh.locate(p,lt,li);
  assert(lt == CDT::VERTEX);
  return face->vertex(li);
}


VertexHandle TriMesh::insert(vec & p){
  return insert(Point(p(0),p(1)));
}

VertexHandle TriMesh::insert(Point p){
  assert(not m_frozen);
  m_dirty = true;
  return m_mesh.insert(p);
}

vec TriMesh::center_of_face(uint fid) const{
  assert(m_frozen);
  uint vid;
  vec point = zeros<vec>(2);
  for(uint i = 0; i < TRI_NUM_VERT; i++){
    vid = m_faces(fid,i);
    point += m_nodes.row(vid).t();
  }
  point /= TRI_NUM_VERT;
  return point; 
}

Point TriMesh::center_of_face(const FaceHandle & face) const{
  // Doesn't need to be frozen
  double x, y;
  x = y = 0;
  VertexHandle vh;
  for(uint i = 0; i < TRI_NUM_VERT; i++){
    vh = face->vertex(0);
    x += vh->point().x();
    y += vh->point().y();
  }
  x /= TRI_NUM_VERT;
  y /= TRI_NUM_VERT;
  return Point(x,y);
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

Points TriMesh::get_spatial_nodes() const{
  assert(m_frozen);
  return m_nodes.head_rows(number_of_vertices());
}

Points TriMesh::get_all_nodes() const{
  assert(m_frozen);
  return m_nodes;
}

Points TriMesh::get_face_centers() const{
  assert(m_frozen);
  uint F = number_of_faces();
  Points centers = Points(F,TRI_NUM_DIM);
  for(uint f = 0; f < F; f++){
    centers.row(f) = center_of_face(f).t();
  }
  return centers;
}


void TriMesh::write_cgal(const string & filename) const{
  assert(m_frozen);
  ofstream fs(filename);
  fs << m_mesh;
  fs.close();
}

void TriMesh::read_cgal(const string & filename){
  assert(!m_frozen);
  m_dirty = true;

  ifstream fs(filename);
  fs >> m_mesh;
  fs.close();
}

void TriMesh::write_shewchuk(string base_filename) const{
  // Write the .node and .ele files. Shewchuk uses these files in Triangle
  // and Stellar
  // Mostly for visualization purposes
  
  assert(m_frozen);
  ofstream node_file,ele_file;
  uint attr = 0; // Number of attributes, will be useful later
  uint bnd = 0; // Boundary marker. Will be important for

  // Regenerate all the supporting information
  
  // Write .node header
  string node_filename = base_filename + ".node";
  node_file.open(node_filename.c_str());
  
  // <# of vertices> <dim> <# of attributes> <# of boundary markers (0 or 1)>
  // NB: ignore the OOB node (node at infinity)
  node_file << (m_nodes.n_rows - 1)
	    << "\t" << TRI_NUM_DIM
	    << "\t" << attr
	    << "\t" << bnd << endl;
  
  // <vertex #> <x> <y> [attributes] [boundary marker]
  // NB: ignore the OOB node (node at infinity)
  for(uint i = 0; i < (m_nodes.n_rows-1); i++){
    node_file << i << "\t" << m_nodes.row(i);
  }
  node_file.close();

  // Write .ele file
  string ele_filename = base_filename + ".ele";
  ele_file.open(ele_filename.c_str());
    // <# of triangles> <nodes per triangle> <# of attributes>
  ele_file << m_faces.n_rows
	   << "\t" << TRI_NUM_VERT
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
  if(m_frozen){
    // Make idempotent
    assert(!m_dirty);
    return;
  }  
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

  // Face cache
  uint f_id = 0;
  for(FaceIterator fit = m_mesh.faces_begin();
      fit != m_mesh.faces_end(); ++fit){
    // Add to face description
    for(uint v = 0; v < TRI_NUM_VERT; v++){
      m_faces(f_id,v) = m_vert_reg[fit->vertex(v)];
    }
    m_face_reg[fit] = f_id++;
  }
  m_dirty = false;
}

vec TriMesh::face_diff(const vec & vertex_function) const{
  assert(m_frozen);

  uint V = number_of_vertices();
  uint F = number_of_faces();
  assert(V == vertex_function.n_elem);
  
  vec diff = vec(F);
  uvec val = uvec(TRI_NUM_VERT);
  VertexHandle vh;
  uint f = 0;
  uint vid = 0;
  for(FaceIterator fit = m_mesh.finite_faces_begin();
      fit != m_mesh.finite_faces_end(); ++fit){
    for(uint v = 0; v < TRI_NUM_VERT; v++){
      vh = fit->vertex(v);
      vid = m_vert_reg.at(vh);
      val(v) = vertex_function(vid);
    }
    diff(f++) = max(val) - min(val);
  }
  assert(f == F);
  return diff; 
}

vec TriMesh::prism_volume(const vec & vertex_function) const{
  // Volumn of a truncated right prism:
  // V = A * (v_0 + v_1 + v_2) / 3.0
  assert(m_frozen);

  uint V = number_of_vertices();
  uint F = number_of_faces();
  assert(V == vertex_function.n_elem);
  
  vec vol = zeros<vec>(F);
  VertexHandle vh;
  uint f = 0;
  uint vid = 0;
  double area,mean_fn;
  CDT::Triangle t;

  
  for(FaceIterator fit = m_mesh.faces_begin();
      fit != m_mesh.faces_end(); ++fit){    
    t = m_mesh.triangle(fit);
    //cout << "Triangle: " << t << endl;
    area = std::abs(t.area());
    //cout << "\tArea: "<< area << endl;;
    mean_fn = 0;
    for(uint v = 0; v < TRI_NUM_VERT; v++){
      vh = fit->vertex(v);
      vid = m_vert_reg.at(vh);
      mean_fn = vertex_function(vid) / (double)TRI_NUM_VERT;
      //cout << "\t f(" << vh->point() << ") = " << vertex_function(vid) << endl;
    }
    //cout << "\tMean function: " << mean_fn << endl;
    //cout << "\tVolume: " << area * mean_fn << endl;
    vol(f++) = area * mean_fn;
  }
  assert(f == F);
  return vol; 
}

mat TriMesh::face_grad(const vec & vertex_function) const{
  assert(m_frozen);

  uint V = number_of_vertices();
  uint F = number_of_faces();
  assert(V == vertex_function.n_elem);
  
  mat grad = mat(F,2);
  VertexHandle vh;
  uint vid;
  vec f = vec(TRI_NUM_VERT);
  mat points = mat(TRI_NUM_VERT,TRI_NUM_DIM);
  mat A = mat(TRI_NUM_DIM,TRI_NUM_DIM);
  vec b = vec(TRI_NUM_DIM);

  uint fid = 0;
  for(FaceIterator fit = m_mesh.finite_faces_begin();
      fit != m_mesh.finite_faces_end(); ++fit){
    // Gather function and point information
    for(uint v = 0; v < TRI_NUM_VERT; v++){
      vh = fit->vertex(v);
      vid = m_vert_reg.at(vh);
      
      f(v) = vertex_function(vid);
      points(v,0) = vh->point().x();
      points(v,1) = vh->point().y();
    }
    // Set up linear equation
    for(uint i = 0; i < TRI_NUM_DIM; i++){
      A.row(i) = points.row(i+1) - points.row(0);
      b(i) = f(i+1) - f(0);
    }

    // Solve linear equation for gradient
    grad.row(fid++) = solve(A,b,solve_opts::fast).t();
  }
  assert(fid == F);
  return grad; 
}

mat TriMesh::find_box_boundary() const{
  assert(m_frozen);
  mat bounds = mat(2,2);

  uint R = m_nodes.n_rows - 1;
  bounds(0,0) = min(m_nodes.col(0).head(R));
  bounds(0,1) = max(m_nodes.col(0).head(R));
  bounds(1,0) = min(m_nodes.col(1).head(R));
  bounds(1,1) = max(m_nodes.col(1).head(R));

  return bounds;
}

void TriMesh::print_vert_reg() const{
  for(VertexRegistry::const_iterator it = m_vert_reg.begin();
      it != m_vert_reg.end(); ++it){
    cout << it->first->point() << ", " << it->second << endl;
  }
}
