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
  assert(2 == B.num_spatial_nodes());
  assert(2 == B.get_spatial_mask().n_elem);
  assert(1 == B.num_special_nodes());
  assert(1 == B.get_special_mask().n_elem);

  
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
  assert(all(uvec{2} == A.get_special_mask()));
  cout << "Passed test_oob." << endl;

  return true;
}

bool test_oob_rand(){
  TypedPoints points = TypedPoints(2*(2*randu<mat>(100,4)-1));
  
  mat bbox1 = mat{
    {-datum::inf,datum::inf},
    {0,1},
    {-datum::inf,datum::inf},
    {0,1}};
  mat bbox2 = mat{
    {0,1},
    {-datum::inf,datum::inf},
    {0,1},
    {-datum::inf,datum::inf}};
  
  TypeRuleList rules;
  rules.emplace_back(new OutOfBoundsRule(bbox1, 1));
  rules.emplace_back(new OutOfBoundsRule(bbox2, 2));
  points.apply_typing_rules(rules);

  mat bbox = mat{{0,1},{0,1},{0,1},{0,1}};
  points.check_in_bbox(bbox);
  assert(100 == points.num_special_nodes() + points.num_spatial_nodes());
  cout << "Passed test_oob_rand." << endl;
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
  
  mat expected = mat{{0.25,-2},{0.5,0.5},{0.25,1}};
  assert(norm(P - expected) < 1e-9);
  cout << "Passed test_wrap_1." << endl;
  return true;
}

bool test_wrap_2(){
  Points P = mat{{-2,-2},{0.5,0.5},{1,1}};
  mat bbox = mat{{-datum::inf,datum::inf},{-1.5,0}};
  WrapRemapper remap = WrapRemapper(bbox);
  remap.remap(P);
  mat expected = mat{{-2,-0.5},{0.5,-1},{1,-0.5}};
  assert(norm(P - expected) < 1e-9);
  cout << "Passed test_wrap_2." << endl;
  return true;
}

bool test_saturate_and_wrap_rand(){
  TypedPoints points = TypedPoints(2 * (2* randu<mat>(100,4) - 1));
  
  mat bbox1 = mat{
    {-datum::inf,datum::inf},
    {0,1},
    {-datum::inf,datum::inf},
    {0,1}};
  mat bbox2 = mat{
    {0,1},
    {-datum::inf,datum::inf},
    {0,1},
    {-datum::inf,datum::inf}};
  
  NodeRemapperList rules;
  rules.emplace_back(new SaturateRemapper(bbox1));
  rules.emplace_back(new WrapRemapper(bbox2));
  points.apply_remappers(rules);

  mat bbox = mat{{0,1},{0,1},{0,1},{0,1}};
  points.check_in_bbox(bbox);
  assert(100 == points.num_special_nodes() + points.num_spatial_nodes());
  cout << "Passed test_oob_rand." << endl;
}


int main(){
  test_build_typed_points();
  test_oob();
  test_oob_rand();
  test_saturate();
  test_wrap_1();
  test_wrap_2();
  test_saturate_and_wrap_rand();
}
