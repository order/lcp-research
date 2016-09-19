#ifndef __Z_TRIMESH_INCLUDED__
#define __Z_TRIMESH_INCLUDED__

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_vertex_base_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/lloyd_optimize_mesh_2.h>
#include <CGAL/Random.h>

#include <map>
#include <fstream>
#include <iostream>
#include <string>

#include <armadillo>

using namespace arma;
using namespace std;

typedef CGAL::Exact_predicates_inexact_constructions_kernel       Kernel;
typedef CGAL::Delaunay_mesh_vertex_base_2<Kernel>                 VertexBase;
typedef CGAL::Delaunay_mesh_face_base_2<Kernel>                   FaceBase;
typedef CGAL::Triangulation_data_structure_2<VertexBase,FaceBase> TDS;
typedef CGAL::Constrained_Delaunay_triangulation_2<Kernel,TDS>    CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT>                  MeshCriteria;
typedef CGAL::Delaunay_mesher_2<CDT, MeshCriteria>                MeshRefiner;

typedef CDT::Vertex_handle   VertexHandle;
typedef CDT::Point           Point;
typedef CDT::Face_handle     FaceHandle;
typedef CDT::Vertex_iterator VertexIterator;
typedef CDT::Face_iterator   FaceIterator;
typedef CDT::Locate_type     LocateType;

typedef map<VertexHandle,uint> VertexRegistry;
typedef map<FaceHandle,uint>   FaceRegistry;

typedef mat    Points;
typedef uvec   Indices;
typedef umat   VertexIndices;
typedef sp_mat ElementDist;
typedef mat    RelDist;

#define TRI_NUM_DIM  2
#define TRI_NUM_VERT 3

Point convert(const vec & point);

struct BaryCoord{
  /*
    Holds the barycentric coordinates. These are the unique weights describing
    a point as a convex combination of vertices in the enclosed face.
    Also indicates if the 
   */
  BaryCoord();
  BaryCoord(bool,uvec,vec);
  
  bool oob;
  uvec indices;
  vec weights;
};
ostream& operator<< (ostream& os, const BaryCoord& coord);


class TriMesh{
  /*
    Basically just a combination of CGAL's constrained Delaunay
    triangular mesh and some registries for indexing faces and vertices.
  */
 public:
  TriMesh();
  TriMesh(const TriMesh &);
  ElementDist points_to_element_dist(const Points &) const;
  ElementDist points_to_element_dist(const Points &,
				     uvec & row_idx_uvec,
				     uvec & col_ptr_uvec,
				     vec & data_vec) const;
  template <typename T> T interpolate(const Points &,
                                      const T&) const;
  
  BaryCoord barycentric_coord(const Point &) const;
  FaceHandle locate_face(const Point &) const;
  VertexHandle locate_vertex(const Point &) const;
  
  VertexHandle insert(vec &);
  VertexHandle insert(Point);

  vec center_of_face(uint fid) const;
  Point center_of_face(const FaceHandle & face) const;
 
  void insert_constraint(VertexHandle & a,
			 VertexHandle & b);
  void refine(double b,double S);
  void lloyd(uint I);

  Points get_spatial_nodes() const;
  Points get_all_nodes() const;
  Points get_face_centers() const;

  uint number_of_faces() const;
  uint number_of_vertices() const;
  uint number_of_nodes() const;
  uint oob_node_index() const;
  
  mat find_bounding_box() const;
  void build_box_boundary(const mat & bbox);
  void build_box_boundary(const vec & lb,
                          const vec & ub);

  vec face_diff(const vec & vertex_function) const;
  vec prism_volume(const vec & vertex_function) const;
  mat face_grad(const vec & vertex_function) const;

  void write_cgal(const string &) const;
  void read_cgal(const string &); 
  void write_shewchuk(string base_filename) const; // To node and ele files

  void freeze();
  void print_vert_reg() const;
  //protected:

  bool m_dirty;
  bool m_frozen;
  
  CDT m_mesh;
  MeshRefiner m_refiner;
  mat m_nodes;
  umat m_faces;
  VertexRegistry m_vert_reg;
  FaceRegistry m_face_reg;

  void regen_caches(); // Derive from m_mesh
};




#endif
