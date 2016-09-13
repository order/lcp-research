#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_3.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <list>
#include <vector>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_3<K>      Triangulation;
typedef Triangulation::Cell_handle    Cell_handle;
typedef Triangulation::Vertex_handle  Vertex_handle;
typedef Triangulation::Locate_type    Locate_type;
typedef Triangulation::Point          Point;

int main(int argc, char ** argv)
{
  assert(3 == argc);
  // construction from a list of points :
  Triangulation T;
  std::ifstream fin(argv[1],std::ios::in);
  fin >> T;
  assert( T.is_valid() ); // checking validity of T
  
  std::ofstream fout(argv[2],std::ios::out);
  fout << T;

  return 0;
}
