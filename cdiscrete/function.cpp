#include "function.h"
#include <assert.h>

InterpFunction::InterpFunction(const vec & val,
			       const RegGrid & grid){
  _val = val;
  _grid = grid;
}

vec InterpFunction::f(const mat & points) const{
  return interp_fn(_val,points,_grid);
}
double InterpFunction::f(const vec & points) const{
  return interp_fn(_val,points.t(),_grid)(0);
}
uint InterpFunction::dom_dim() const{
  return _grid.low.n_elem;
}

InterpMultiFunction::InterpMultiFunction(const mat & val,
					 const RegGrid & grid){
  _val = val;
  _grid = grid;
}

mat InterpMultiFunction::f(const mat & points) const{
  return interp_fns(_val,points,_grid);
}
vec InterpMultiFunction::f(const vec & points) const{
  mat R = interp_fns(_val,points.t(),_grid);
  assert(R.n_rows == 1);
  return interp_fns(_val,points.t(),_grid).row(0);
}
uint InterpMultiFunction::dom_dim() const{
  return _grid.low.n_elem;
}
uint InterpMultiFunction::range_dim() const{
  return _val.n_cols;
}

ConstMultiFunction::ConstMultiFunction(uint n, double val){
  _val = val;
  _n = n;
}

mat ConstMultiFunction::f(const mat & points) const{
  uint N = points.n_rows;  
  return mat(N,_n).fill(_val);
}
vec ConstMultiFunction::f(const vec & points) const{
  return vec(_n).fill(_val);
}
uint ConstMultiFunction::dom_dim() const{
  return -1; // Not sure; doesn't matter
}
uint ConstMultiFunction::range_dim() const{
  return _n;
}

ProbFunction::ProbFunction(MultiFunction * base_fn){
  _base = base_fn;
}

mat ProbFunction::f(const mat & points) const{
  uint N = points.n_rows;
  mat unnorm = _base->f(points);
  assert(all(all(unnorm >= 0)));
  
  vec Z = sum(unnorm,1);
  assert(N == Z.n_elem);
  for(uint i = 0; i < N; i++){
    unnorm.row(i) /= Z(i);
  }
  return unnorm;
  
}
vec ProbFunction::f(const vec & points) const{
  mat r = f(conv_to<mat>::from(points));
  assert(1 == r.n_rows);
  return r.row(0);
}
uint ProbFunction::dom_dim() const{
  return _base->dom_dim(); // Not sure; doesn't matter
}
uint ProbFunction::range_dim() const{
  return _base->range_dim();
}
