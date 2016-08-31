#define CGAL_MESH_2_OPTIMIZER_VERBOSE

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

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_mesh_vertex_base_2<K>                Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K>                  Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>        Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds>  CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT>            Criteria;
typedef CGAL::Delaunay_mesher_2<CDT, Criteria>              Mesher;

typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Point Point;
typedef CDT::Face_handle Face_handle;
typedef CDT::Vertex_iterator Vertex_iterator;
typedef CDT::Face_iterator Face_iterator;

using namespace arma;
using namespace std;



void write_2D_node_and_ele_files(const CDT & cdt,string base_filename){
  // Write the .node and .ele files. Shewchuk uses these files in Triangle
  // and Stellar

  ofstream node_file,ele_file;
  uint dim = 2;
  uint attr = 0; // Number of attributes, will be useful later
  uint bnd = 0; // Boundary marker. Will be important for 

  // Write .node header
  // <# of vertices> <dimension (must be 2)> <# of attributes> <# of boundary markers (0 or 1)>
  string node_filename = base_filename + ".node";
  node_file.open(node_filename.c_str());
  node_file << "# Generated from CGAL" << endl;
  node_file << cdt.number_of_vertices() << "\t" << dim << "\t" << attr << "\t" << bnd << endl;  
  
  // Index vertices
  uint vertex_id = 0;
  map<Vertex_handle,uint> vertex_id_map;
  for(Vertex_iterator vit = cdt.vertices_begin();
      vit != cdt.vertices_end(); ++vit){
    // <vertex #> <x> <y> [attributes] [boundary marker]
    node_file << vertex_id << "\t" << vit->point().x() << "\t" << vit->point().y()<< endl;;
    vertex_id_map[vit] = vertex_id++;
  }
  node_file.close();

  // Write .ele file
  // <# of triangles> <nodes per triangle> <# of attributes>
  string ele_filename = base_filename + ".ele";
  ele_file.open(ele_filename.c_str());
  ele_file << "# Generated from CGAL" << endl;
  ele_file << cdt.number_of_faces() << "\t" << (dim + 1) << "\t" << attr << endl;
  
  uint face_id = 0;
  for(Face_iterator fit = cdt.faces_begin();
      fit != cdt.faces_end(); ++fit){
    ele_file << face_id++;
    for(uint v = 0; v < dim + 1; v++){
      ele_file << "\t" << vertex_id_map[fit->vertex(v)];
    }
    ele_file << endl;
  }
  ele_file.close();
}

void add_bang_bang_curves(CDT & cdt,
			  Vertex_handle & v_zero,
			  Vertex_handle & v_upper_left,
			  Vertex_handle & v_lower_right,
			  uint num_curve_points){
  Vertex_handle v_old = v_zero;
  Vertex_handle v_new;
  double x,y;
  double N = num_curve_points;
  // -ve x, +ve y
  for(double i = 1; i < N; i++){
    y = i / N; // Uniform over y
    x = - y * y;
    v_new = cdt.insert(Point(x,y));
    cdt.insert_constraint(v_old,v_new);
    v_old = v_new;
  }
  cdt.insert_constraint(v_old,v_upper_left);

  v_old = v_zero;
  for(double i = 1; i < N; i++){
    y = -i / N;
    x = y * y;
    v_new = cdt.insert(Point(x,y));
    cdt.insert_constraint(v_old,v_new);
    v_old = v_new;
  }
  cdt.insert_constraint(v_old,v_lower_right);
}

vec barycentric_coords(const CDT & cdt,
		       vec & p){
  assert(2 == point.n_elem);

  double x = p(0);
  double y = p(1);
  
  // Locate face
  Point cgal_point = Point(x,y);
  Face_handle face = cdt.locate(cgal_point);

  // Extract vertices
  vec X = vec(3);
  vec Y = vec(3);
  for(uint i = 0; i < 3; i++){
    X(i) = face->vertex(i)->point().x();
    Y(i) = face->vertex(i)->point().y();
  }
  

  vec c = vec(3);
  double Det = (Y(1) - Y(2))*(X(0) - X(2)) + (X(2) - X(1))*(Y(0) - Y(2));
  c(0) = ((Y(1) - Y(2))*(x - X(2)) + (X(2) - X(1))*(y - Y(2))) / Det;
  c(1) = ((Y(2) - Y(0))*(x - X(2)) + (X(0) - X(2))*(y - Y(2))) / Det;
  c(2) = 1.0 - c(0) - c(1);

  vec recon = vec(2);
  recon(0) = dot(X,c);
  recon(1) = dot(Y,c);
  assert(approx_equal(recon,p,"reldiff",1e-12));
  
  return c;
}

int main()
{
  CDT cdt;
  
  Vertex_handle v_low_left = cdt.insert(Point(-1,-1));
  Vertex_handle v_low_right = cdt.insert(Point(1,-1));
  Vertex_handle v_up_left = cdt.insert(Point(-1,1));
  Vertex_handle v_up_right = cdt.insert(Point(1,1));
  cdt.insert_constraint(v_low_left, v_low_right);
  cdt.insert_constraint(v_low_left, v_up_left);
  cdt.insert_constraint(v_up_right, v_low_right);
  cdt.insert_constraint(v_up_right, v_up_left);

  Vertex_handle v_zero = cdt.insert(Point(0,0));
  uint num_curve_points = 25;
  add_bang_bang_curves(cdt,v_zero,v_up_left,v_low_right,num_curve_points);
  
  cout << "Number of vertices: " << cdt.number_of_vertices() << endl;
  cout << "Refining triangulation..." << endl;
  Mesher mesher(cdt);
  mesher.refine_mesh();
  mesher.set_criteria(Criteria(0.125,0.2));
  mesher.refine_mesh();

  cout << "Optimizing placement..." << endl;
  CGAL::lloyd_optimize_mesh_2(cdt,
    CGAL::parameters::max_iteration_number = 50);
  cout << "\tNumber of vertices: " << cdt.number_of_vertices() << endl;
  cout << "\tNumber of faces: " << cdt.number_of_faces() << endl;

  write_2D_node_and_ele_files(cdt,"test");
}
