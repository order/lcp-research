#include "function.h"
#include <assert.h>
RealFunction::~RealFunction(){};

MultiFunction::~MultiFunction(){};

InterpFunction::InterpFunction(const vec & val,
			       const RegGrid & grid){
  _val = vec(val);
  _grid = grid;
}
InterpFunction::~InterpFunction(){
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
  _val = mat(val);
  _grid.low = grid.low;
  _grid.high = grid.high;
  _grid.num_cells = grid.num_cells;
  
  uint G = num_grid_points(grid);
  assert(G == val.n_rows);
}

InterpMultiFunction::~InterpMultiFunction(){
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

ProbFunction::ProbFunction(const mat & val,
			   const RegGrid & grid) : InterpMultiFunction(val,grid)
{}

mat ProbFunction::f(const mat & points) const{
  mat Z = interp_fns(_val,points,_grid);
  uint N = Z.n_rows;
  uint d = Z.n_cols;
  for(uint i = 0; i < N; i++){
    double norm = sum(Z.row(i));
    if(norm > 0){
      Z.row(i) /= sum(Z.row(i));
    }
    else{
      Z.row(i).fill(1.0 / d);
    }
  }
  return Z;
}
vec ProbFunction::f(const vec & point) const{
  mat points = conv_to<mat>::from(point.t());
  mat r = f(points);
  assert(1 == r.n_rows);
  return r.row(0).t();
}
