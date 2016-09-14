#ifndef __Z_TETMESH_INCLUDED__
#define __Z_TETMESH_INCLUDED__

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_3.h>

#include <map>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <armadillo>

#define TET_NUM_VERT 4
#define TET_NUM_DIM 3
#define ALMOST_ZERO 1e-15
#define PRETTY_SMALL 1e-8

using namespace arma;
using namespace std;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_3<K>     Triangulation;
typedef Triangulation::Cell_handle   CellHandle;
typedef Triangulation::Vertex_handle VertexHandle;
typedef Triangulation::Locate_type   LocateType;
typedef Triangulation::Point         Point;

typedef Triangulation::Finite_cells_iterator    CellIterator;
typedef Triangulation::Finite_vertices_iterator VertexIterator;

typedef map<VertexHandle,uint> VertexRegistry;
typedef map<CellHandle,uint>   CellRegistry;

typedef mat    Points;
typedef uvec   Indices;
typedef umat   VertexIndices;
typedef sp_mat ElementDist;
typedef mat    RelDist;

typedef arma::vec::fixed<TET_NUM_DIM> VertexVec;
typedef arma::mat::fixed<TET_NUM_VERT,TET_NUM_DIM> VertexMat;
typedef arma::vec::fixed<TET_NUM_VERT> CoordVec;
typedef arma::uvec::fixed<TET_NUM_VERT> TetVertIndexVec;

// Conversion routines
VertexVec point_to_vertex_vec(const Point & point);
template<typename T> Point vec_to_point(const T & point);
VertexMat cell_to_vertex_mat(const CellHandle & tet);
template<typename T> bool is_asc_sorted(const T & v);

struct TetBaryCoord{
  /*
    Holds the barycentric coordinates. These are the unique weights describing
    a point as a convex combination of vertices in the enclosed face.
    Also indicates if the 
   */
  TetBaryCoord();
  TetBaryCoord(bool,TetVertIndexVec,CoordVec);
  
  bool oob;
  TetVertIndexVec indices;
  CoordVec weights;
};
ostream& operator<< (ostream& os, const TetBaryCoord& coord);


class TetMesh{
  /*
    Basically just a combination of CGAL's constrained Delaunay
    triangular mesh and some registries for indexing faces and vertices.
  */
 public:
  TetMesh();
  TetMesh(const TetMesh &);
  ElementDist points_to_element_dist(const Points &) const;
  ElementDist points_to_element_dist(const Points &,
				     uvec & row_idx_uvec,
				     uvec & col_ptr_uvec,
				     vec & data_vec) const;
  template <typename T> T interpolate(const Points &,
                                      const T&) const;
  
  TetBaryCoord barycentric_coord(const Point &) const;
  
  VertexHandle insert(const VertexVec &);
  VertexHandle insert(const Point&);

  VertexMat get_vertex_mat(uint tet_id) const;

  VertexVec center_of_cell(uint tet_id) const;

  Points get_spatial_nodes() const;
  Points get_all_nodes() const;
  Points get_cell_centers() const;

  uint number_of_cells() const;
  uint number_of_vertices() const;
  uint number_of_nodes() const;
  uint oob_node_index() const;
  mat find_box_boundary() const;

  //vec face_diff(const vec & vertex_function) const;
  //vec prism_volume(const vec & vertex_function) const;
  //mat face_grad(const vec & vertex_function) const;

  void write_cgal(const string &) const;
  void read_cgal(const string &); 
  void write_shewchuk(string base_filename) const; // To node and ele files

  void freeze();
  
  
  //protected:

  bool m_dirty;
  bool m_frozen;
  
  Triangulation m_mesh;
  mat m_nodes;
  umat m_cells;
  VertexRegistry m_vert_reg;
  CellRegistry m_cell_reg;

  void regen_caches(); // Derive from m_mesh
};


#endif
