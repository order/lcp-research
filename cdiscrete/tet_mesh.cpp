#include "tet_mesh.h"
#include "misc.h"
#include "io.h"
#include "car.h"

////////////////////////////////////////////////
// CGAL <-> Armadillo conversion routines

VertexVec point_to_vertex_vec(const Point & p){
  return VertexVec {p[0],p[1],p[2]};
}

template<typename T> Point vec_to_point(const T & point){
  assert(TET_NUM_DIM == point.n_elem);
  return Point(point[0],point[1],point[2]);
}
template Point vec_to_point<VertexVec>(const VertexVec & point);
template Point vec_to_point<vec>(const vec & point);

VertexMat tet_to_vertex_mat(const CellHandle & tet){
  VertexMat vertex_mat;
  for(uint v = 0; v < TET_NUM_VERT; v++){
    for(uint d = 0; d < TET_NUM_DIM; d++){
      vertex_mat(v,d) = tet->vertex(v)->point()[d];
    }
  }
  return vertex_mat;
}

//////////////////////////////////////////////////
// Barycentric coord structure constructors
TetBaryCoord::TetBaryCoord():oob(true){}
TetBaryCoord::TetBaryCoord(bool o, TetVertIndexVec i, CoordVec w) :
  oob(o),indices(i),weights(w){}

ostream& operator<< (ostream& os, const TetBaryCoord& coord){
  if(coord.oob){
    os << "OOB" << endl;
  }
  else{
    os << coord.indices.t() << coord.weights.t();
  }
  return os;
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
  uint M = number_of_nodes();

  Point p;
  TetBaryCoord coord;

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

template <typename T> T TetMesh::interpolate(const Points & points,
                                             const T& data) const{
  assert(m_frozen);
  
  uint N = points.n_rows;
  uint d = points.n_cols;
  assert(TET_NUM_DIM == d);
  
  uint NN = number_of_nodes();
  assert(data.n_rows == NN); // Should include oob info
  
  ElementDist dist = points_to_element_dist(points);
  assert(size(dist) == size(NN,N));

  T ret = dist.t() * data;
  assert(ret.n_rows == N);
  assert(ret.n_cols == data.n_cols);
  return ret;
}
template mat TetMesh::interpolate<mat>(const Points &, const mat&) const;
template vec TetMesh::interpolate<vec>(const Points &, const vec&) const;

TetBaryCoord TetMesh::barycentric_coord(const Point & point) const{  
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
    return TetBaryCoord(); // Out of bounds
  }

  uint tet_id = m_cell_reg.at(tet);
  TetVertIndexVec vert_idx = m_cells.row(tet_id).t();
  assert(vert_idx.is_sorted());
  VertexMat V = get_vertex_mat(tet_id);

  // Build the barycentric coordinate system
  VertexVec v0 = V.row(0).t();
  arma::mat::fixed<TET_NUM_DIM,TET_NUM_DIM> T;
  T = V.tail_rows(TET_NUM_DIM).t();
  T = T.each_col() - v0;

  // Solve the barycentric system
  VertexVec pvec = point_to_vertex_vec(point);
  arma::vec::fixed<TET_NUM_DIM> partial_coords = arma::solve(T,pvec-v0);

  // Solution only gives the last 3 components of the coordinate system
  // First component is 1 - the sum of the last three
  double agg = arma::sum(partial_coords);
  CoordVec coords;
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
  VertexVec recon = V.t() * coords;
  if(accu(abs(recon-pvec)) > PRETTY_SMALL){
    cout << "Abs reconstruction error: " << accu(abs(recon-pvec)) << endl;
    assert(approx_equal(recon,pvec,"absdiff",PRETTY_SMALL));
  }  
  return TetBaryCoord(false,vert_idx,coords);
}

VertexHandle TetMesh::insert(const VertexVec & pvec){
  assert(not m_frozen);
  m_dirty = true;
  return insert(Point(pvec(0),pvec(1),pvec(2)));
}

VertexHandle TetMesh::insert(const Point & p){
  assert(not m_frozen);
  m_dirty = true;
  return m_mesh.insert(p);
}

VertexMat TetMesh::get_vertex_mat(uint tet_id) const{
  assert(m_frozen);
  uvec vert_idx = m_cells.row(tet_id).t();
  assert(TET_NUM_VERT == vert_idx.n_elem);
  assert(vert_idx.is_sorted());
  
  VertexMat V;
  for(uint v = 0; v < TET_NUM_VERT; v++){
    V.row(v) = m_nodes.row(vert_idx(v));
  }
  return V;
}

VertexVec TetMesh::center_of_cell(uint tet_id) const{
  assert(m_frozen);

  VertexMat V = get_vertex_mat(tet_id);
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

void TetMesh::write_cgal(const string & filename) const{
  assert(m_frozen);
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
  return m_mesh.number_of_cells();
}
uint TetMesh::number_of_vertices() const{
  return m_mesh.number_of_vertices();
}

uint TetMesh::number_of_nodes() const{
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
  m_nodes = mat(m_mesh.number_of_vertices()+1,TET_NUM_DIM);
  m_cells = umat(m_mesh.number_of_cells(),TET_NUM_VERT);
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

  // Sort the vertex indices.
  m_cells = sort(m_cells,"ascend",1);
  
  m_dirty = false;
}

mat TetMesh::find_box_boundary() const{
  assert(m_frozen);
  mat bounds = mat(TET_NUM_DIM,2);

  uint spatial_nodes = number_of_vertices();
  for(uint d = 0; d < TET_NUM_DIM; d++){
    bounds(d,0) = min(m_nodes.col(d).head(spatial_nodes));
    bounds(d,1) = max(m_nodes.col(d).head(spatial_nodes));
  }
  return bounds;
}

