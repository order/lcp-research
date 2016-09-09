#include <iostream>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Simple_cartesian.h>

#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>

#include <CGAL/Polyhedral_mesh_domain_3.h>

#include <CGAL/make_mesh_3.h>
#include <CGAL/convex_hull_3.h>

#include <armadillo>



// Domain
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
//typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3 Point;

typedef CGAL::Polyhedron_3<K> Polyhedron;
typedef CGAL::Polyhedral_mesh_domain_3<Polyhedron,K> Mesh_domain;

// Triangulation
#ifdef CGAL_CONCURRENT_MESH_3
  typedef CGAL::Mesh_triangulation_3<
    Mesh_domain,
    CGAL::Kernel_traits<Mesh_domain>::Kernel, // Same as sequential
    CGAL::Parallel_tag                        // Tag to activate parallelism
  >::type Tr;
#else
  typedef CGAL::Mesh_triangulation_3<Mesh_domain>::type Tr;
#endif

typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;
// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;
// To avoid verbose function and named parameters call
using namespace CGAL::parameters;
using namespace arma;
void build_rect(const vec & lb,
                const vec & ub,
                Polyhedron & poly){
  
  std::vector<Point> points;
  vec p = vec(3);
  Point P;
  for(uint b = 0; b < pow(2,3); b++){
    for(uint i = 0; i < 3; i++){
      if(b & (1 << i)){
        p(i) = ub(i);
      }
      else{
        p(i) = lb(i);
      }
    }
    P = Point(p(0),p(1),p(2));
    points.push_back(P);
  }
  CGAL::convex_hull_3(points.begin(), points.end(), poly);  
}

int main()
{
  Polyhedron poly1;
  vec lb = -1 * ones<vec>(3);
  vec ub = 1 * ones<vec>(3);
  build_rect(lb,ub,poly1);

  Polyhedron poly2;
  lb = zeros<vec>(3);
  ub = 2 * ones<vec>(3);
  build_rect(lb,ub,poly2);

  
  // Domain (Warning: Sphere_3 constructor uses squared radius !)
  Mesh_domain domain(poly1);
  // Mesh criteria
  Mesh_criteria criteria(facet_angle=30, facet_size=0.15, facet_distance=0.025,
                         cell_radius_edge_ratio=2, cell_size=0.15);  
  // Mesh generation
  C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria);
  std::cout << "Number of cells: " << c3t3.number_of_cells() << std::endl;
  std::cout << "Number of facets: " << c3t3.number_of_facets() << std::endl;

  // Output
  std::ofstream medit_file("out.mesh");
  c3t3.output_to_medit(medit_file);
  return 0;
}
