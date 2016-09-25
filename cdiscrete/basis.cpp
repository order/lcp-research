#include <assert.h>

#include "basis.h"
#include "misc.h"

mat make_ring_basis(const Points & points,
               uint S){
  
  vec dist = lp_norm(points,2,1);

  uint V = points.n_rows;
  mat basis = zeros<mat>(V,S);
  double l,u;
  double inc_rad = 1.0 / (double) S;
  uvec mask;
  for(uint s = 0; s < S; s++){
    l = inc_rad * s;
    u = inc_rad * (s + 1);
    if(s == S-1) u += 1e-12;
    mask = find(dist >= l and dist < u);
    if(0 == mask.n_elem)
      basis.col(s).randu();
    else
      basis(mask, uvec{s}).fill(1);
  }
  basis = orth(basis);
  return basis;
}

vec make_spike_basis(const Points & points,
                const vec & center){
  uint V = points.n_rows;
  vec dist = lp_norm(points.each_row() - center.t(),2,1);
  double upper_rad = max(dist);
  double lower_rad = 0;
  double rad;
  uint count;
  cout << "Finding spike radius: " << endl;
  while(true){
    rad = (upper_rad + lower_rad) / 2.0;
    count = (conv_to<uvec>::from(find(dist <= rad))).n_elem;
    cout << "\t At r=" << rad << ", "
         << count << " nodes included" << endl;
    assert(count <= V);

    if(count == 3){
      break;
    }
    if(count < 3){
      lower_rad = rad;
    }
    else{
      upper_rad = rad;
    }
  }
  
  vec basis = zeros<vec>(V);
  basis(find(dist <= rad)).fill(1);
  basis /= norm(basis,2);
  assert(V == basis.n_elem);
  return basis;
}

mat make_grid_basis(const Points & points,
               const mat & bbox,
               uint X, uint Y){
  uint V = points.n_rows;
  double dx = (bbox(0,1) - bbox(0,0)) / (double)X;
  double dy = (bbox(1,1) - bbox(1,0)) / (double)Y;
  mat basis = zeros<mat>(V,X*Y);
  uint b = 0;
  uvec mask;
  double xl,yl,xh,yh;
  for(uint i = 0; i < X;i++){
    assert(b < X*Y);
    xl = (double)i*dx + bbox(0,0);
    xh = xl + dx;
    if(i == X-1) xh += 1e-12;
    for(uint j = 0; j < Y; j++){
      yl = (double)j*dy + bbox(1,0);
      yh = yl + dy;
      if(j == Y-1) yh += 1e-12;
      mask = find(points.col(0) >= xl
                  and points.col(0) < xh
                  and points.col(1) >= yl
                  and points.col(1) < yh);
      basis(mask,uvec{b}).fill(1);
      b++;
    }
  }
  basis = orth(basis);
  return basis;
}

mat make_voronoi_basis(const Points & points,
                       const Points & centers){
  uint N = points.n_rows;
  uint C = centers.n_rows;
  assert(points.n_cols == centers.n_cols);
  
  mat dist = mat(N,C);
  for(uint c = 0; c < C; c++){
    dist.col(c) = lp_norm(points.each_row() - centers.row(c),2,1);
  }
  uvec partition = col_argmin(dist);

  mat basis = zeros<mat>(N,C);
  for(uint c = 0; c < C; c++){
    basis(find(partition == c),uvec{c}).fill(1);
  }

  return basis;
}

mat make_radial_fourier_basis(const Points & points,
                        uint K,double max_freq){
  uint N = points.n_rows;
  mat basis = mat(N,2*K+1);

  vec r = lp_norm(points,2,1);
  double omega;
  for(uint k = 0; k < K; k++){
    omega = ((double)k+1.0) * max_freq / (double) K;
    omega *= 2.0*datum::pi;
    basis.col(K-k-1) = sin(omega*r);
    basis.col(K+k+1) = cos(omega*r);
  }
  basis.col(K).fill(1/sqrt(N));
  return basis;
}
