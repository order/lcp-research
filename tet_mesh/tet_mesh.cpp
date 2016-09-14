#include "tet_mesh.h"

////////////////////////////////////////////////
// CGAL <-> Armadillo conversion routines

VertexVec point_to_vertex_vect(const Point & p){
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

template<typename T> bool is_asc_sorted(const T & v){
  uint N = v.n_elem;
  for(uint i = 0; i < N-1;i++){
    if(v[i] > v[i+1]) return false;
  }
  return true;
}
template bool is_asc_sorted<TetVertIndexVec>(const TetVertIndexVec &);
template bool is_asc_sorted<uvec>(const uvec &);

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
  m_mesh(),,m_dirty(true),m_frozen(false){}

TetMesh::TetMesh(const TriMesh & other) :
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
      assert(0 == coord.indices.n_elem);
      assert(0 == coord.weights.n_elem);     
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
template mat TriMesh::interpolate<mat>(const Points &, const mat&) const;
template vec TriMesh::interpolate<vec>(const Points &, const vec&) const;

TetBaryCoord TetMesh::barycentric_coord(const Point & point) const{  
  assert(m_frozen);
  // Locate face
  int li,lj;
  LocateType loc_type;
  CellHandle tet = m_mesh.locate(point,loc_type,li,lj);
  // If the location type isn't a face of some kind,
  // or is an infinite cell, then it's out of bounds.
  if(loc == CDT::OUTSIDE_CONVEX_HULL
     or loc == CDT::OUTSIDE_AFFINE_HULL
     or m_mesh.is_infinite(tet)){
    return BaryCoord(true,uvec(),vec()); // Out of bounds
  }

  uint tet_id = m_cell_reg[tet];
  TetVertIndexVec vert_idx = m_cells[tet_id];
  assert(is_asc_sorted(vert_idx));
  VertMat V = get_vertex_mat(tet_id);

  // Build the barycentric coordinate system
  VertexVec v0 = V.row(0).t();
  arma::mat::fixed<TET_NUM_DIM,TET_NUM_DIM> T;
  T = V.tail_rows(TET_NUM_DIM).t();
  T = T.each_col() - v0;

  // Solve the barycentric system
  VertexVec pvec = point_to_vertex_vec(point);
  arma::mat::fixed<TET_NUM_DIM> partial_coords = arma::solve(T,pvec-v0);

  // Solution only gives the last 3 components of the coordinate system
  // First component is 1 - the sum of the last three
  double agg = arma::sum(partial_coords);
  CoordVec coords;
  coords(0) = 1.0 - agg;
  coords.tail(3) = partial_coords;
  assert(np.all(coords >= 0));
  assert(np.all(coords <= 1));
  
  // Check the reconstruction error
  VertexVec recon = V.t() * C;
  if(accu(abs(recon-pvec)) > PRETTY_SMALL){
    cout << "Abs reconstruction error: " << accu(abs(recon-pvec)) << endl;
    assert(approx_equal(recon,pvec,"absdiff",PRETTY_SMALL));
  }  
  return TetBaryCoord(false,vert_idx,coords);
}

VertexHandle TetMesh::insert(const VertexVec & pvec){
  assert(not m_frozen);
  m_dirty = true;
  return insert(Point(p(0),p(1),p(2)));
}

VertexHandle TetMesh::insert(Point p){
  assert(not m_frozen);
  m_dirty = true;
  return m_mesh.insert(p);
}

VertexMat TetMesh::get_vertex_mat(uint tet_id) const{
  assert(m_frozen);
  TetVertIndexVec vert_idx = conv_to<TetVertIndexVec>::from(m_cells.row(tet_id));
  VertexMat V;
  for(uint v = 0; v < TET_NUM_VERT; v++){
    V.row(v) = m_nodes.row(vert_idx(v));
  }
  return V;
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


///////////////////////////////////////////////
// OLD CODE ///////////////////////////////////
///////////////////////////////////////////////

/*
  Calculation the barycentric coordinates
*/
CoordVec barycentric(const VertexVec q, VertexMat V){
  VertexVec v0 = V.row(0).t();
  arma::mat::fixed<TET_NUM_DIM,TET_NUM_DIM> T;
  T = V.tail_rows(TET_NUM_DIM).t();
  T = T.each_col() - v0;

  // Unroll this solve?
  arma::vec::fixed<TET_NUM_DIM> c = arma::solve(T,q-v0);
  
  double agg = arma::sum(c);
  assert(agg <= 1.0);
  assert(agg >= 0.0);

  CoordVec C;
  C(0) = 1.0 - agg;
  C.tail(3) = c;
  return C;
}

void locate(const Point & p,
            const Triangulation & tets){
  int li,lj;
  Locate_type lt;
  Cell_handle tet = tets.locate(p,lt,li,lj);

  std::cout << "Looking for " << p << std::endl;
  
  if(lt == Triangulation::OUTSIDE_CONVEX_HULL
     or lt == Triangulation::OUTSIDE_AFFINE_HULL){
    std::cout << "Outside hull of points." << std::endl;
    return;
  }
  if(tets.is_infinite(tet)){
    std::cout << "Out of bounds." << std::endl;
    return;
  }

  VertexVec pvec = point_to_vertex_vect(p);
  VertexMat vmat = tet_to_vertex_mat(tet);
  std::cout << "Found in cell:\n" << vmat;
  CoordVec cvec =  barycentric(pvec,vmat);
  std::cout << "Barycentric:\n" << cvec.t();
  std::cout << "Reconstruct:\n" << cvec.t() * vmat;
}

int main(int argc, char ** argv)
{
  if(3 != argc){
    std::cerr << "Usage: tet_mesh [input file] [output file]" << std::endl;
    return -1;
  }
  
  assert(3 == argc);
  // construction from a list of points :
  /*std::list<Point> L;
  for(uint b = 0; b < 8; b++){
    Point p(b & 1,(b & 2) >> 1,(b & 4) >> 2);
    L.push_front(p);
  }    
  Triangulation tet_mesh(L.begin(), L.end());*/
  
  Triangulation tet_mesh;
  std::ifstream fin(argv[1],std::ios::in);
  if(!fin.is_open()){
    std::cerr << argv[1] << " did not open successfully." << std::endl;
    return -1;
  }
  fin >> tet_mesh;
  if(!tet_mesh.is_valid()){
    std::cerr << "tet_mesh invalid. Something messed up with the input file?"
              << std::endl;
    tet_mesh.is_valid(true); // Verbose recheck
  }

  std::cout << "Number of vertices: "
            << tet_mesh.number_of_vertices() << std::endl;
  std::cout << "Number of finite cells: "
            << tet_mesh.number_of_finite_cells() << std::endl;
  std::cout << "Number of cells: "
            << tet_mesh.number_of_cells() << std::endl;

  Point x = Point(0.5,0.5,0.5);
  locate(x,tet_mesh);
  
  std::ofstream fout(argv[2],std::ios::out);
  if(!fout.is_open()){
    std::cerr << argv[2] << " did not open successfully." << std::endl;
    return -1;
  }
  fout << tet_mesh;

  return 0;
}
