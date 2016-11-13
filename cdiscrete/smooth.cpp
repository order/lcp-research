#include "smooth.h"
#include "basis.h"

sp_mat gaussian_smoother(const Points & points,
                         const double bandwidth,
                         const double zero_thresh){
  uint N = points.n_rows;

  vector<tuple<uint,uint,double> > triples;
  for(uint col = 0; col < N; col++){
    vec g = gaussian(points,points.row(col).t(),bandwidth);
    g /= accu(g);
    uvec idx = find(g > zero_thresh); // Threshold
    
    for(uint i = 0; i < idx.n_elem; i++){
      uint row = idx(i);
      double value = g(row);
      triples.push_back(make_tuple(row,col,value));
    }
  }

  uint nnz = triples.size();
  umat loc = umat(2,nnz);
  vec data = vec(nnz);
  for(uint i = 0; i < nnz; i++){
    loc(0,i) = get<0>(triples[i]);
    loc(1,i) = get<1>(triples[i]);
    data(i) = get<2>(triples[i]);
  }

  return sp_mat(loc,data,N,N);
}
