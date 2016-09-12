#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_3.h>
#include <CGAL/convex_hull_3.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <list>
#include <set>

#include <armadillo>
#include <boost/algorithm/string.hpp>    
#include <boost/algorithm/string/split.hpp>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_3<K>      Triangulation;
typedef Triangulation::Finite_vertices_iterator Finite_vertices_iterator;
typedef Triangulation::Finite_edges_iterator Finite_edges_iterator;
typedef Triangulation::Finite_facets_iterator Finite_facets_iterator;
typedef Triangulation::Finite_cells_iterator Finite_cells_iterator;
typedef Triangulation::Simplex        Simplex;
typedef Triangulation::Locate_type    Locate_type;
typedef Triangulation::Point          Point;

using namespace arma;
using namespace std;

mat read_vertices(const string & filename){
  /*
    A pretty brittle parser for INRIA .mesh files
    Extract just the vertex information.
   */
  ifstream fin(filename);
  string line;
  // Find the vertex section
  bool found = false;
  while(getline(fin,line)){
    boost::algorithm::to_lower(line);
    if(std::string::npos != line.find("vertices")){
      found = true;
      break;
    }
  }
  assert(found);
  getline(fin,line);
  uint num_vert = stoul(line);

  mat points = mat(num_vert,3);
  std::vector<std::string> tokens;
  for(uint i; i < num_vert; i++){
    getline(fin,line);
    cout << line << endl;
    boost::trim(line);
    boost::split(tokens, line, boost::is_any_of(" \t"),boost::token_compress_on);
    cout << tokens.size() << endl;
    assert(4 == tokens.size());
    for(uint j = 0; j < 3; ++j){
      points(i,j) = stod(tokens[j]);
    }
  }
  return points;
}

int main()
{
  // construction from a list of points :
  std::list<Point> L;
  L.push_front(Point(0,0,0));
  L.push_front(Point(1,0,0));
  L.push_front(Point(0,1,0));
  L.push_front(Point(0,1,1));
  L.push_front(Point(1,1,0));
  Triangulation T(L.begin(), L.end());

  cout << T;
  
  return 0;
}
