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

#define NUMDIM  2
#define NUMVERT 3

#define ALMOST_ZERO 1e-15

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
  ElementDist points_to_element_dist(const Points &);  
  ElementDist points_to_element_dist(const Points &,
				     uvec & row_idx_uvec,
				     uvec & col_ptr_uvec,
				     vec & data_vec);
  BaryCoord barycentric_coord(const Point &);

  FaceHandle locate(const Point &) const;
  
  VertexHandle insert(vec &);
  VertexHandle insert(Point);
  void insert_constraint(VertexHandle & a,
			 VertexHandle & b);
  void refine(double b,double S);
  void lloyd(uint I);

  Points get_spatial_nodes();
  Points get_all_nodes();

  uint number_of_faces() const;
  uint number_of_vertices() const;
  uint number_of_nodes() const;
  uint oob_node_index() const;

  void freeze();
  
  void write(string base_filename); // To node and ele files
  
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


void saturate(Points & points,
	      const vec &lb,
	      const vec &ub);
// Subsume when merge with cdiscrete
mat make_points(const vector<vec> & grids);

// DI stuff
void add_di_bang_bang_curves(TriMesh & mesh,
			     VertexHandle & v_zero,
			     VertexHandle & v_upper_left,
			     VertexHandle & v_lower_right,
			     uint num_curve_points);

Points double_integrator(const Points & points,
			 double a,double t);

mat build_di_costs(const Points & points);
vec build_di_state_weights(const Points & points);
#endif