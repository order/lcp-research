#include "tet_mesh.h"
#include "misc.h"
#include "io.h"

using namespace tet_mesh;

////////////////////////////////////////////////
// CGAL <-> Armadillo conversion routines

vec tet_mesh::point_to_vertex_vec(const Point & p){
  return vec {p[0],p[1],p[2]};
}

template<typename T> Point tet_mesh::vec_to_point(const T & point){
  assert(TET_NUM_DIM == point.n_elem);
  return Point(point[0],point[1],point[2]);
}
template Point tet_mesh::vec_to_point<vec>(const vec & point);

mat tet_mesh::tet_to_vertex_mat(const CellHandle & tet){
  mat vertex_mat;
  for(uint v = 0; v < TET_NUM_VERT; v++){
    for(uint d = 0; d < TET_NUM_DIM; d++){
      vertex_mat(v,d) = tet->vertex(v)->point()[d];
    }
  }
  return vertex_mat;
}

/////////////////////////////////////////////////
// Tet mesh code

TetMesh::TetMesh() :
  m_mesh(),m_dirty(true),m_frozen(false){}

TetMesh::TetMesh(const TetMesh & other) :
  m_mesh(other.m_mesh),m_dirty(true),m_frozen(false){
  regen_caches();
}

ElementDist TetMesh::points_to_element_dist(const Points & points) const{
  // Ignore CSC returns
  uvec row = uvec();
  uvec col = uvec();
  vec data = vec();
  return points_to_element_dist(points,row,col,data);
}

ElementDist TetMesh::points_to_element_dist(const Points & points,
					    uvec & row_idx_uvec,
					    uvec & col_ptr_uvec,
					    vec & data_vec) const{
  assert(TET_NUM_DIM == points.n_cols);
  assert(m_frozen);

  vector<uword> row_idx;
  vector<uword> col_ptr;
  vector<double> data;

  uint N = points.n_rows;
  uint oob_idx = oob_node_index();
  uint M = number_of_all_nodes();

  Point p;
  BaryCoord coord;

  // Set up sparse matrix via row indices and col pointers.
  // Assume elements are visited in col sorted order,
  // Column i is described by the row indices and data in the index range
  // col_ptr[i],...,col_ptr[i+1]-1 (empty if col_ptr[i] == col_ptr[i+1])

  uint oob_count;
  for(uint i = 0; i < N; i++){
    p = Point(points(i,0),points(i,1),points(i,2));
    coord = barycentric_coord(p);
    
    // Start new column
    // First element of new column located at
    // the current length of the row index vec
    col_ptr.push_back(row_idx.size());

    if(coord.oob){
      // Out of bounds; all weight on oob node
      row_idx.push_back(oob_idx);
      data.push_back(1.0);
      oob_count++;
    }
    else{
      // In bounds; add barycentric coords.
      assert(TET_NUM_VERT == coord.indices.n_elem);
      assert(TET_NUM_VERT == coord.weights.n_elem);
      
      for(uint v = 0; v < TET_NUM_VERT; v++){
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

template <typename T> T TetMesh::base_interpolate(const Points & points,
                                                  const T& data) const{
  assert(m_frozen);
  
  uint N = points.n_rows;
  uint d = points.n_cols;
  assert(TET_NUM_DIM == d);
  
  uint NN = number_of_all_nodes();
  assert(data.n_rows == NN); // Should include oob info
  
  ElementDist dist = points_to_element_dist(points);
  assert(size(dist) == size(NN,N));

  T ret = dist.t() * data;
  assert(ret.n_rows == N);
  assert(ret.n_cols == data.n_cols);
  return ret;
}
vec TetMesh::interpolate(const Points & points, const vec & data) const{
  return base_interpolate<vec>(points,data);
}
mat TetMesh::interpolate(const Points & points, const mat & data) const{
  return base_interpolate<mat>(points,data);
}

BaryCoord TetMesh::barycentric_coord(const Point & point) const{  
  assert(m_frozen);
  // Locate cell
  int li,lj;
  LocateType loc;
  CellHandle tet = m_mesh.locate(point,loc,li,lj);
  // If the location type isn't a face of some kind,
  // or is an infinite cell, then it's out of bounds.
  if(loc == Triangulation::OUTSIDE_CONVEX_HULL
     or loc == Triangulation::OUTSIDE_AFFINE_HULL
     or m_mesh.is_infinite(tet)){
    return BaryCoord(true,uvec(),vec()); // Out of bounds
  }

  uint tet_id = m_cell_reg.at(tet);
  umat vert_idx = m_cells.row(tet_id).t();
  assert(vert_idx.is_sorted());
  mat V = get_vertex_mat(tet_id);

  // Build the barycentric coordinate system
  vec v0 = V.row(0).t();
  mat T;
  T = V.tail_rows(TET_NUM_DIM).t();
  T = T.each_col() - v0;

  // Solve the barycentric system
  vec pvec = point_to_vertex_vec(point);
  vec partial_coords = arma::solve(T,pvec-v0);

  // Solution only gives the last 3 components of the coordinate system
  // First component is 1 - the sum of the last three
  double agg = arma::sum(partial_coords);
  vec coords = vec(TET_NUM_VERT);
  coords(0) = 1.0 - agg;
  coords.tail(3) = partial_coords;
  
  // Get rid of tiny entries (for sparsity & numerical reasons)
  coords(find(abs(coords) < ALMOST_ZERO)).fill(0);
  coords /= accu(coords); // Rescale to be convex
  if(not all(coords >= 0)){
    cerr << "[ERR] Point: " << point
         << "\n\tCoords: " << coords.t()
         << "\tVertex mat:\n"<< V;
  }
  assert(all(coords >= 0));
  assert(all(coords <= 1));
  
  // Check the reconstruction error
  vec recon = V.t() * coords;
  if(accu(abs(recon-pvec)) > PRETTY_SMALL){
    cout << "Abs reconstruction error: " << accu(abs(recon-pvec)) << endl;
    assert(approx_equal(recon,pvec,"absdiff",PRETTY_SMALL));
  }  
  return BaryCoord(false,vert_idx,coords);
}

VertexHandle TetMesh::insert(const vec & pvec){
  assert(not m_frozen);
  m_dirty = true;
  return insert(Point(pvec(0),pvec(1),pvec(2)));
}

VertexHandle TetMesh::insert(const Point & p){
  assert(not m_frozen);
  m_dirty = true;
  return m_mesh.insert(p);
}

mat TetMesh::get_vertex_mat(uint tet_id) const{
  assert(m_frozen);
  uvec vert_idx = m_cells.row(tet_id).t();
  assert(TET_NUM_VERT == vert_idx.n_elem);
  assert(vert_idx.is_sorted());
  uint V = number_of_spatial_nodes();
  
  mat vert_mat = mat(TET_NUM_VERT,TET_NUM_DIM);
  uint v_id;
  for(uint v = 0; v < TET_NUM_VERT; v++){
    v_id = vert_idx(v);
    assert(v_id < V);
    vert_mat.row(v) = m_nodes.row(v_id);
  }
  return vert_mat;
}

vec TetMesh::center_of_cell(uint tet_id) const{
  assert(m_frozen);

  mat V = get_vertex_mat(tet_id);
  return sum(V,0).t() / TET_NUM_VERT;
}

Points TetMesh::get_spatial_nodes() const{
  assert(m_frozen);
  return m_nodes.head_rows(number_of_vertices());
}

Points TetMesh::get_all_nodes() const{
  assert(m_frozen);
  return m_nodes;
}

Points TetMesh::get_cell_centers() const{
  assert(m_frozen);
  uint C = number_of_cells();
  Points centers = Points(C,TET_NUM_DIM);
  for(uint c = 0; c < C; c++){
    centers.row(c) = center_of_cell(c).t();
  }
  return centers;
}
umat TetMesh::get_cell_node_indices() const{
  return m_cells;
}

mat TetMesh::find_bounding_box() const{
  assert(m_frozen);
  mat bounds = mat(TET_NUM_DIM,2);

  uint spatial_nodes = number_of_vertices();
  for(uint d = 0; d < TET_NUM_DIM; d++){
    bounds(d,0) = min(m_nodes.col(d).head(spatial_nodes));
    bounds(d,1) = max(m_nodes.col(d).head(spatial_nodes));
  }
  return bounds;
}

vec TetMesh::prism_volume(const vec & vertex_function) const{
  // Volumn of a truncated right prism:
  // V = A * (v_0 + v_1 + v_2) / 3.0
  assert(m_frozen);

  uint V = number_of_vertices();
  uint C = number_of_cells();
  assert(V == vertex_function.n_elem);
  
  vec volume = zeros<vec>(C);
  VertexHandle vh;
  uint c = 0;
  uint vid = 0;
  double vol,mean_fn;
  Triangulation::Tetrahedron t;
  
  for(CellIterator cit = m_mesh.finite_cells_begin();
      cit != m_mesh.finite_cells_end(); ++cit){    
    t = m_mesh.tetrahedron(cit);
    vol = std::abs(t.volume());
    mean_fn = 0;
    for(uint v = 0; v < TET_NUM_VERT; v++){
      vh = cit->vertex(v);
      vid = m_vert_reg.at(vh);
      mean_fn = vertex_function(vid) / (double)TET_NUM_VERT;
    }
    volume(c++) = vol * mean_fn;
  }
  assert(c == C);
  return volume; 
}
vec TetMesh::cell_volume() const{
  assert(m_frozen);
  
  uint C = number_of_cells();  
  vec volume = zeros<vec>(C);
  Triangulation::Tetrahedron t;

  uint c = 0;
  for(CellIterator cit = m_mesh.finite_cells_begin();
      cit != m_mesh.finite_cells_end(); ++cit){    
    t = m_mesh.tetrahedron(cit);
    volume(c++) = std::abs(t.volume());
  }
  assert(c == C);
  return volume; 
}

mat TetMesh::cell_gradient(const vec & vertex_function) const{
  assert(m_frozen);

  uint V = number_of_vertices();
  uint C = number_of_cells();
  assert(V == vertex_function.n_elem);
  
  mat grad = mat(C,TET_NUM_DIM);
  VertexHandle vh;
  uint vid;
  vec f = vec(TET_NUM_VERT);
  mat points = mat(TET_NUM_VERT,TET_NUM_DIM);
  mat A = mat(TET_NUM_DIM,TET_NUM_DIM);
  vec b = vec(TET_NUM_DIM);

  uint cid = 0;
  for(CellIterator cit = m_mesh.finite_cells_begin();
      cit != m_mesh.finite_cells_end(); ++cit){   
    // Gather function and point information
    for(uint v = 0; v < TET_NUM_VERT; v++){
      vh = cit->vertex(v);
      vid = m_vert_reg.at(vh);
      f(v) = vertex_function(vid);
      for(uint d = 0; d < TET_NUM_DIM; d++){
        points(v,d) = vh->point()[d];
      }
    }
    // Set up linear equation
    for(uint i = 0; i < TET_NUM_DIM; i++){
      A.row(i) = points.row(i+1) - points.row(0);
      b(i) = f(i+1) - f(0);
    }
    // Solve linear equation for gradient
    grad.row(cid++) = solve(A,b,solve_opts::fast).t();
  }
  assert(cid == C);
  return grad; 
}


void TetMesh::write_cgal(const string & filename) const{
  assert(m_frozen);
  cout << "Writing to " << filename << endl
       <<"\tVertices: " << number_of_vertices() << endl
       <<"\tTetrahedra: " << number_of_cells() << endl;
  ofstream fs(filename);
  fs << m_mesh;
  fs.close();
}

void TetMesh::read_cgal(const string & filename){
  assert(!m_frozen);
  m_dirty = true;

  ifstream fs(filename);
  fs >> m_mesh;
  
  assert(m_mesh.is_valid());
  fs.close();
}

uint TetMesh::number_of_cells() const{
  return m_mesh.number_of_finite_cells();
}
uint TetMesh::number_of_vertices() const{
  return m_mesh.number_of_vertices();
}

uint TetMesh::number_of_spatial_nodes() const{
  return m_mesh.number_of_vertices();
}

uint TetMesh::number_of_all_nodes() const{
  // Number of spatial vertices + 1 oob node
  return number_of_vertices() + 1;
} 

uint TetMesh::oob_node_index() const{
  /*
    Only the one oob node so far
   */
  return m_mesh.number_of_vertices();
}

void TetMesh::freeze(){
  if(m_frozen){
    // Make idempotent
    assert(!m_dirty);
    return;
  }  
  regen_caches();
  m_frozen = true;
}

void TetMesh::regen_caches(){
  if(!m_dirty) return;
  assert(not m_frozen);
  
  // Regenerate cell and vertex caches
  uint V = number_of_all_nodes();
  uint C = number_of_cells();
  m_nodes = mat(V,TET_NUM_DIM);
  m_cells = umat(C,TET_NUM_VERT);
  m_vert_reg.clear();
  m_cell_reg.clear();

  uint v_id = 0;
  for(VertexIterator vit = m_mesh.finite_vertices_begin();
      vit != m_mesh.finite_vertices_end(); ++vit){
    // Add to point mat
    for(uint d = 0; d < TET_NUM_DIM; d++){
      m_nodes(v_id,d) = vit->point()[d];
    }
    // Register
    m_vert_reg[vit] = v_id++;
  }
  // Set OOB node.
  for(uint d = 0; d < TET_NUM_DIM; d++){
    m_nodes(v_id,d) = HUGE_VAL;
  }
  assert(v_id == (V-1));


  // Cell cache
  uint c_id = 0;
  for(CellIterator cit = m_mesh.finite_cells_begin();
      cit != m_mesh.finite_cells_end(); ++cit){
    
    // Get the vertices in each cell
    for(uint v = 0; v < TET_NUM_VERT; v++){
      v_id = m_vert_reg[cit->vertex(v)]; // Get the id for the vth vertex
      m_cells(c_id,v) = v_id;
    }
    // Register this cell
    m_cell_reg[cit] = c_id++;
  }
  assert(c_id == C);

  // Sort the vertex indices.
  m_cells = sort(m_cells,"ascend",1);
  
  m_dirty = false;
}


