#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>

#include <CGAL/Polyhedral_mesh_domain_3.h>
#include <CGAL/refine_mesh_3.h>
#include <CGAL/make_mesh_3.h>

#include <CGAL/convex_hull_3.h>

#include <vector>
#include <iostream>
#include <fstream>
// IO
#include <CGAL/IO/Triangulation_geomview_ostream_3.h>

// Domain
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point;
typedef CGAL::Polyhedron_3<K> Polyhedron;
typedef CGAL::Polyhedral_mesh_domain_3<Polyhedron, K> Mesh_domain;

// Triangulation
typedef CGAL::Mesh_triangulation_3<Mesh_domain>::type Tr;
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;
// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;
// To avoid verbose function and named parameters call
using namespace CGAL::parameters;

/*
  Function for building a rectangular prism.
  This 
*/
void build_rect(const std::vector<double> & lb,
                const std::vector<double> ub,
                Polyhedron & poly){
  
std::vector<Point> points;
  double p[3];
  Point P;
  for(uint b = 0; b < pow(2,3); b++){
    for(uint i = 0; i < 3; i++){
      if(b & (1 << i)){
        p[i] = ub[i];
      }
      else{
        p[i] = lb[i];
      }
    }
    P = Point(p[0],p[1],p[2]);
    points.push_back(P);
  }
  CGAL::convex_hull_3(points.begin(), points.end(), poly);  
}

int main()
{
  std::vector<double> lb;
  std::vector<double> ub;
  for(uint d = 0; d < 3; d++){
    lb.push_back(-1.0);
    ub.push_back(1.0);
  }
  Polyhedron poly;
  build_rect(lb,ub,poly);
  
  // Define functions
  // Domain (Warning: Sphere_3 constructor uses square radius !)
  Mesh_domain domain(poly);
  // Set mesh criteria
  Mesh_criteria criteria(facet_angle=25, facet_size=0.15, facet_distance=1e-2,
                         cell_radius_edge_ratio=3);
  // Mesh generation
  C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria);
  
  // Mesh output
  std::ofstream medit_file("out.mesh");
  c3t3.output_to_medit(medit_file);

  // Triangulation output and input
  std::cout << c3t3.triangulation();
  C3t3 c3t3_read_back;
  std::cin >> c3t3_read_back.triangulation();
  //NB: file compiles when these last three lines are commented out.
 
  return 0;
}
