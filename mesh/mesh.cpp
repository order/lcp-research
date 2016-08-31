#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Random.h>

#include <iostream>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;
typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Point Point;
typedef CDT::Face_handle Face_handle;
int main()
{
  CDT cdt;
  Vertex_handle va = cdt.insert(Point(-4,0));
  Vertex_handle vb = cdt.insert(Point(0,-1));
  Vertex_handle vc = cdt.insert(Point(4,0));
  Vertex_handle vd = cdt.insert(Point(0,1));
  cdt.insert(Point(2, 0.6));
  cdt.insert_constraint(va, vb);
  cdt.insert_constraint(vb, vc);
  cdt.insert_constraint(vc, vd);
  cdt.insert_constraint(vd, va);
  std::cout << "Number of vertices: " << cdt.number_of_vertices() << std::endl;
  std::cout << "Refining triangulation..." << std::endl;
  CGAL::refine_Delaunay_mesh_2(cdt, Criteria());
  std::cout << "\tNumber of vertices: " << cdt.number_of_vertices() << std::endl;
  std::cout << "Futher refining triangulation..." << std::endl;
  CGAL::refine_Delaunay_mesh_2(cdt, Criteria(0.125, 0.25));
  std::cout << "\tNumber of vertices: " << cdt.number_of_vertices() << std::endl; 

  for(uint i = 0; i < 10;i++){
    cdt.insert(Point(CGAL::default_random.get_double(-4,4),
		     CGAL::default_random.get_double(-4,4)));
  }
  CGAL::refine_Delaunay_mesh_2(cdt, Criteria(0.125, 0.5));
  std::cout << "Number of vertices: " << cdt.number_of_vertices() << std::endl;

  Point query = Point(1,1);
  Face_handle f = cdt.locate(query);
  std::cout << "Face index: " << f->vertex(0)->point() << std::endl;
  std::cout << "Face index: " << f->vertex(1)->point() << std::endl;
  std::cout << "Face index: " << f->vertex(2)->point() << std::endl;

}
