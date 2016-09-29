#include <assert.h>

#include "basis.h"
#include "misc.h"
#include <set>

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

sp_mat make_sample_basis(uint N,
                        uint K){
  sp_mat basis = sp_mat(N,K);
  set<uword> keys;
  uvec samples = randi<uvec>(K,distr_param(0,N-1));
  for(uint k = 0; k < K; k++){
    while(keys.count(samples(k)) > 0){
      samples(k) = randi<uvec>(1,distr_param(0,N-1))(0);
    }
    basis(samples(k),k) = 1;
    keys.insert(samples(k));
  }
  return basis;
}

mat make_rbf_basis(const Points & points,
                   const Points & centers,
                   double bandwidth){
  uint N = points.n_rows;
  uint K = centers.n_rows;
  mat basis = zeros<mat>(N,K);
  double value_thresh = 1e-8;
  double dist_thresh = std::sqrt(-bandwidth * std::log(value_thresh));
  for(uint k = 0; k < K; k++){
    vec dist = lp_norm(points.each_row() - centers.row(k),2,1);
    uvec idx = find(dist <= dist_thresh);
    basis(idx,uvec{k}) = exp(-dist(idx)/bandwidth);
  }
  basis = orth(basis);
  return basis;
}

double find_radius(const vec & dist,
                   const bvec & mask,
                   uint target){
  uint N = mask.n_elem;
  assert(N == dist.n_elem);
  assert(target > 0);
  
  if(N - sum(mask) <= target){
    return max(dist);
  }
  double u = max(dist(find(0 == mask)));
  double l = 0;
  double m;
  uvec idx;
  while(u > l + 1e-3){
    m = (u + l) / 2.0;
    idx = find(dist(find(0 == mask)) <= m);
    if(target == idx.n_elem)
      return m; // Just right
    else if (target < idx.n_elem)
      u = m; // Too many
    else
      l = m; // Too few
  }
  return u; // Can't pip the target; better to return too many.
}

sp_mat make_ball_basis(const Points & points,
                       const Points & centers,
                       uint R){
  /*
    Make a set of balls.
    Each ball has unique points via set subtraction
    (think "moon shape" if two balls overlap)
    Each ball should have exactly R points, but there may be
    Fewer if we run out of points.
   */
  uint N = points.n_rows;
  uint K = centers.n_rows;
  sp_mat basis = sp_mat(N,K);

  bvec mask = zeros<bvec>(N);
  vec dist;
  uvec idx;
  double r;
  uint k;
  for(k = 0; k < K; k++){
    dist = lp_norm(points.each_row() - centers.row(k),2,1);
    r = find_radius(dist,mask,R);
    idx = find(dist < r);
    for(uint i = 0; i < idx.n_elem; i++){
      // Paint all elements of idx that aren't
      // already in another basis
      if(0 == mask(idx(i)))
        basis(idx(i),k) = 1.0 / (double) idx.n_elem;
    }
    
    mask(idx).fill(1); // Add to mask
    basis.col(k) /= norm(basis.col(k),2);
    if(N == sum(mask)){
      break;
    }
  }
  basis.resize(N,k); // Avoid all zero
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

IndexPartition voronoi_partition(const Points & points,
                                 const Points & centers){
  uint N = points.n_rows;
  uint K = centers.n_rows;
  assert(points.n_cols == centers.n_cols);
  
  mat dist = mat(N,K);  
  for(uint k = 0; k < K; k++){
    dist.col(k) = lp_norm(points.each_row() - centers.row(k),2,1);
  }
  uvec P = col_argmin(dist); // partition index

  // Assign indices to partitions
  IndexPartition partition;
  partition.resize(K); 
  for(uint k = 0; k < K; k++){
    uvec idx = find(P == k);
    for(uint i = 0; i < idx.n_elem; i++){
      partition[k].insert(idx(i));
    }
  }

  // Remove empty bases
  uint c = 0;
  while(c < partition.size()){
    if(0 == partition[c].size())
      partition.erase(partition.begin() + c);
    else
      c++;
  }
  return partition;
}

set<uint> ball_indices(const Points & points,
                       const vec & center,
                       uint R){
    vec dist = lp_norm(points.each_row() - center,2,1);
    double r = find_radius(dist,R);
    uvec idx = find(dist < r);

    set<uint> basis;
    for(uint i = 0; i < idx.n_elem; i++){
      basis.insert(idx(i));
    }
}

void add_basis(IndexPartition & partition,
                         const set<uint> & basis){
  for(IndexIterator it = partition.begin();
      it != partition.end(); it++){
    vector<uint> overlap;
    set_intersection(it->begin(),it->end(),
                     basis.begin(),basis.end(),
                     back_inserter(overlap));
    for(vector<uint>::iterator oit = overlap.begin();
        oit != overlap.end(); oit++){
      it->erase(*oit);
    }
  }
  partition.push_back(basis);
}

sp_mat build_basis_from_partition(const IndexPartition & partition,uint N){
  uint K = partition.size();
  sp_mat basis = sp_mat(N,K);
  for(uint k = 0; k < K; k++){
    double v = 1.0 * sqrt(partition[k].size());
    for(set<uint>::const_iterator it = partition[k].begin();
        it != partition[k].end(); it++){
      basis(*it,k) = v;
    }
  }
  return basis;
}
