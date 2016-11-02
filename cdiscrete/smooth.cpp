#include "smooth.h"
sp_mat gaussian_smoother(const Points & points,
                         const double bandwidth,
                         const double zero_thresh){
  uint N = points.n_rows;

  vector<tuple<uint,uint,double> > triples;
  for(uint col = 0; col < N; col++){
    vec g = gaussian(points,points.row(col).t(),bandwidth);
    uvec idx = find(g > zero_thresh); // Threshold
    
    for(uint i = 0; i < idx.n_elem; i++){
      uint row = idx(j);
      double value = g(row);
      triples.pushback(make_tuple(row,col,value));
    }
  }

  uint nnz = triples.size();
  mat loc = mat(2,nnz);
  vec data = vec(nnz);
  for(uint i = 0; i < nnz; i++){
    loc(0,i) = triples[i].get(0);
    loc(1,i) = triples[i].get(1);
    data(i) = triples[i].get(2);
  }

  return sp_mat(loc,data,N,N);
}
