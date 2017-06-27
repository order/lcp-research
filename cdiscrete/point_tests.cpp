#include <assert.h>

#include <armadillo>
#include "points.h"

using namespace arma;
using namespace std;

bool test_build_typed_points(){
  TypedPoints A = TypedPoints(mat{{-2,-2},{0.5,0.5},{1,1}});
  assert(3 == A.num_spatial_nodes());
  assert(3 == A.get_spatial_mask().n_elem);
  assert(0 == A.num_special_nodes());
  assert(0 == A.get_special_mask().n_elem);
  
  TypeRegistry reg = TypeRegistry();
  reg[1] = 1;
  TypedPoints B = TypedPoints(mat{{-2,-2},{0.5,0.5},{1,1}},reg);
  assert(2 == A.get_spatial_mask().n_elem);
  assert(1 == A.num_special_nodes());
  
  assert(B.m_points.has_nan());
  assert(is_finite(B.m_points.rows(uvec{0,2})));
  cout << "Passed test_build_typed_points." << endl;
  return true;
}

bool test_oob(){
  TypedPoints A = TypedPoints(mat{{2,-1},{-1e9,0.5},{-100,2}});
  assert(3 == A.num_spatial_nodes());
  assert(0 == A.num_special_nodes());
  
  mat bbox = mat{{-datum::inf,2},{-1,1}};
  OutOfBoundsRule rule = OutOfBoundsRule(bbox, 1);

  A.apply_typing_rule(rule);
  assert(1 == A.num_special_nodes());
  assert(2 == A.num_spatial_nodes());
  assert(all(uvec{2} == A.get_spatial_mask()));

  return true;
}

bool test_saturate(){
  Points P = mat{{-2,-2},{0.5,0.5},{1,1}};
  mat bbox = mat{{-1,1},{-1,1}};
  SaturateRemapper remap = SaturateRemapper(bbox);
  remap.remap(P);
  
  mat expected = mat{{-1,-1},{0.5,0.5},{1,1}};
  assert(norm(P - expected) < 1e-9);
  cout << "Passed test_saturate." << endl;
  return true;
}

bool test_wrap_1(){
  Points P = mat{{-2,-2},{0.5,0.5},{1,1}};
  mat bbox = mat{{0,0.75},{-datum::inf,datum::inf}};
  WrapRemapper remap = WrapRemapper(bbox);
  remap.remap(P);
  
  mat expected = mat{{0.5,-2},{0.5,0.5},{0.25,1}};
  assert(norm(P - expected) < 1e-9);
  cout << "Passed test_wrap_1." << endl;
  return true;
}

bool test_wrap_2(){
  Points P = mat{{-2,-2},{0.5,0.5},{1,1}};
  mat bbox = mat{{-datum::inf,datum::inf},{-1.5,0}};
  WrapRemapper remap = WrapRemapper(bbox);
  remap.remap(P);

  mat expected = mat{{-2,-0.5},{0.5,-0.5},{1,-0.5}};
  assert(norm(P - expected) < 1e-9);
  cout << "Passed test_wrap_2." << endl;
  return true;
}


int main(){
  test_saturate();
  test_build_typed_points();
  test_oob();
  test_wrap_1();
  test_wrap_2();
}
