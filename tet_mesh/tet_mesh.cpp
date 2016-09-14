#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_3.h>

#include <armadillo>

#include <iostream>
#include <fstream>
#include <cassert>
#include <list>
#include <vector>

#define TET_NUM_VERT 4
#define TET_NUM_DIM 3


typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_3<K>      Triangulation;
typedef Triangulation::Cell_handle    Cell_handle;
typedef Triangulation::Cell_iterator  Cell_iterator;
typedef Triangulation::Vertex_handle  Vertex_handle;
typedef Triangulation::Locate_type    Locate_type;
typedef Triangulation::Point          Point;

typedef arma::vec::fixed<TET_NUM_DIM> VertexVec;
typedef arma::vec::fixed<TET_NUM_VERT> CoordVec;
typedef arma::mat::fixed<TET_NUM_VERT,TET_NUM_DIM> VertexMat;

VertexMat tet_to_vertex_mat(const Cell_handle & tet){
  VertexMat vertex_mat;
  for(uint v = 0; v < TET_NUM_VERT; v++){
    for(uint d = 0; d < TET_NUM_DIM; d++){
      vertex_mat(v,d) = tet->vertex(v)->point()[d];
    }
  }
  return vertex_mat;
}

VertexVec point_to_vertex_vect(const Point & p){
  VertexVec v;
  for(uint d = 0; d < TET_NUM_DIM; d++){
    v(d) = p[d];
  }
  return v;
}

CoordVec barycentric(const VertexVec q, VertexMat V){
  arma::mat::fixed<TET_NUM_DIM,TET_NUM_DIM> T;
  VertexVec v0 = V.row(0).t();
  
  T = V.tail_rows(TET_NUM_DIM).t();
  T = T.each_col() - v0;

  arma::vec::fixed<TET_NUM_DIM> c = arma::solve(T,q-v0);
  double agg = arma::sum(c);
  assert(agg <= 1.0);
  assert(agg >= 0.0);

  CoordVec C;
  C.head(3) = c;
  C(3) = 1.0 - agg;
  return C;
}

void locate(const Point & p,
            const Triangulation & tets){
  int li,lj;
  Locate_type lt;
  Cell_handle tet = tets.locate(p,lt,li,lj);

  std::cout << "Looking for " << p << std::endl;
  
  if(lt == Triangulation::OUTSIDE_CONVEX_HULL
     or lt == Triangulation::OUTSIDE_AFFINE_HULL){
    std::cout << "Outside hull of points." << std::endl;
    return;
  }
  if(tets.is_infinite(tet)){
    std::cout << "Out of bounds." << std::endl;
    return;
  }

  VertexVec pvec = point_to_vertex_vect(p);
  VertexMat vmat = tet_to_vertex_mat(tet);
  std::cout << "Found in cell:\n" << vmat;
  CoordVec cvec =  barycentric(pvec,vmat);
  std::cout << "Barycentric:\n" << cvec.t();
  std::cout << "Reconstruct:\n" << cvec.t() * vmat;
}

int main(int argc, char ** argv)
{
  if(3 != argc){
    std::cerr << "Usage: tet_mesh [input file] [output file]" << std::endl;
    return -1;
  }
  
  assert(3 == argc);
  // construction from a list of points :
  /*std::list<Point> L;
  for(uint b = 0; b < 8; b++){
    Point p(b & 1,(b & 2) >> 1,(b & 4) >> 2);
    L.push_front(p);
  }    
  Triangulation tet_mesh(L.begin(), L.end());*/
  
  Triangulation tet_mesh;
  std::ifstream fin(argv[1],std::ios::in);
  if(!fin.is_open()){
    std::cerr << argv[1] << " did not open successfully." << std::endl;
    return -1;
  }
  fin >> tet_mesh;
  if(!tet_mesh.is_valid()){
    std::cerr << "tet_mesh invalid. Something messed up with the input file?"
              << std::endl;
    tet_mesh.is_valid(true); // Verbose recheck
  }

  std::cout << "Number of vertices: "
            << tet_mesh.number_of_vertices() << std::endl;
  std::cout << "Number of finite cells: "
            << tet_mesh.number_of_finite_cells() << std::endl;
  std::cout << "Number of cells: "
            << tet_mesh.number_of_cells() << std::endl;

  Point x = Point(0.5,0.5,0.5);
  locate(x,tet_mesh);
  
  std::ofstream fout(argv[2],std::ios::out);
  if(!fout.is_open()){
    std::cerr << argv[2] << " did not open successfully." << std::endl;
    return -1;
  }
  fout << tet_mesh;

  return 0;
}
