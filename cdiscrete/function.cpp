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
  _grid.low = grid.low;
  _grid.high = grid.high;
  _grid.num_cells = grid.num_cells;
  
  uint G = num_grid_points(grid);
  assert(G == val.n_rows);
}

mat InterpMultiFunction::f(const mat & points) const{
  return interp_fns(_val,points,_grid);
}
vec InterpMultiFunction::f(const vec & point) const{
  mat points = conv_to<mat>::from(point.t());
  mat R = interp_fns(_val,points,_grid);
  assert(1 == R.n_rows);
  return R.row(0).t();
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
vec ProbFunction::f(const vec & point) const{
  mat points = conv_to<mat>::from(point.t());
  assert(1 == points.n_rows);
  assert(point.n_elem == points.n_cols);
  
  mat res = f(points);
  assert(1 == res.n_rows);
  assert(res.n_cols == range_dim());
  return res.row(0).t();
}
uint ProbFunction::dom_dim() const{
  return _base->dom_dim(); // Not sure; doesn't matter
}
uint ProbFunction::range_dim() const{
  return _base->range_dim();
}
