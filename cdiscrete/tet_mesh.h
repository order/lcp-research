#ifndef __Z_TETMESH_INCLUDED__
#define __Z_TETMESH_INCLUDED__

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_3.h>

#include <map>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "discretizer.h"

#include <armadillo>

using namespace arma;
using namespace std;

namespace tet_mesh{

  typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
  typedef CGAL::Triangulation_3<Kernel>                Triangulation;
  typedef Triangulation::Cell_handle                   CellHandle;
  typedef Triangulation::Vertex_handle                 VertexHandle;
  typedef Triangulation::Locate_type                   LocateType;
  typedef Triangulation::Point                         Point;

  typedef Triangulation::Finite_cells_iterator    CellIterator;
  typedef Triangulation::Finite_vertices_iterator VertexIterator;

  typedef map<VertexHandle,uint> VertexRegistry;
  typedef map<CellHandle,uint>   CellRegistry;

  typedef mat    Points;
  typedef uvec   Indices;
  typedef umat   VertexIndices;
  typedef sp_mat ElementDist;

  const static int TET_NUM_DIM =  3;
  const static int TET_NUM_VERT = 4;

  // Conversion routines
  vec point_to_vertex_vec(const Point & point);
  template<typename T> Point vec_to_point(const T & point);
  mat tet_to_vertex_mat(const CellHandle & tet);

  class TetMesh : public Discretizer{
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
    template <typename T> T base_interpolate(const Points &,
                                             const T&) const;
    vec interpolate(const Points &, const vec&) const;
    mat interpolate(const Points &, const mat&) const;
  
    BaryCoord barycentric_coord(const Point &) const;
  
    VertexHandle insert(const vec &);
    VertexHandle insert(const Point&);

    mat get_vertex_mat(uint tet_id) const;

    vec center_of_cell(uint tet_id) const;

    Points get_spatial_nodes() const;
    Points get_all_nodes() const;
    Points get_cell_centers() const;
    umat get_cell_node_indices() const;

    uint number_of_cells() const;
    uint number_of_vertices() const;
    uint number_of_spatial_nodes() const;
    uint number_of_all_nodes() const;
    uint oob_node_index() const;
    mat find_bounding_box() const;

    vec prism_volume(const vec & vertex_function) const;
    vec cell_volume() const;
    mat cell_gradient(const vec & value) const;

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

}
#endif
