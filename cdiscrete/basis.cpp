#include <assert.h>
#include "basis.h"
#include "misc.h"

vec gabor_wavelet(const vec & points,
                  const double center,
                  const double freq,
                  const double bandwidth,
                  const double shift){
  // 1D version
  return gabor_wavelet(points,
                       vec{center},
                       vec{freq},
                       bandwidth,
                       shift);
}

vec gabor_wavelet(const Points & points,
                  const vec & center,
                  const vec & freq,
                  const double bandwidth,
                  const double shift){
  uint N = points.n_rows;
  uint D = points.n_cols;

  assert(D == center.n_elem);
  assert(D == freq.n_elem);
  
  // Gabor wavelet (real valued frequencies only)
  vec sqdist = sum(pow(points.each_row() - center.t(),2),1);
  vec gauss = exp(-bandwidth*sqdist);
  vec wave = sin(2.0 * datum::pi * (points * freq) + shift);
  
  return gauss % wave;
}
vec gaussian(const vec & points,
             const double center,
             const double bandwidth){
  return gaussian(points,
                  vec{center},
                  bandwidth);
}
vec gaussian(const Points & points,
             const vec & center,
             const double bandwidth){
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(D == center.n_elem);
  
  // Gabor wavelet (real valued frequencies only)
  vec gauss = exp(-bandwidth * sum(pow(points.each_row() - center.t(),2),1));
  return gauss;
}

double find_radius(const vec & dist,
                   uint target){
  uint N = dist.n_rows;
  bvec mask = zeros<bvec>(N);
  return find_radius(dist,mask,target);
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


mat dist_mat(const Points & A, const Points & B){
  assert(A.n_cols == B.n_cols);
  uint N = A.n_rows;
  uint M = B.n_rows;
  mat dist = mat(N,M);  
  for(uint i = 0; i < M; i++){
    dist.col(i) = lp_norm(A.each_row() - B.row(i),2,1);
  }
  return dist;
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
  assert(K == accu(basis));
  return basis; // Should be orthonormal by default
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
    if(N == sum(mask)){
      break;
    }
  }
  basis.resize(N,k); // Avoid all zero
  basis = sp_normalise(basis,2,0); // Should be almost ortho
  return basis;
}

mat make_rbf_basis(const Points & points,
                   const Points & centers,
                   double bandwidth,
                   double cutoff_thresh){
  uint N = points.n_rows;
  uint K = centers.n_rows;
  
  mat basis = zeros<mat>(N,K+1);
  basis.col(K) = ones<vec>(N);
  for(uint k = 0; k < K; k++){
    basis.col(k) = gaussian(points,centers.row(k).t(),bandwidth);
  }
  //basis(find(basis < cutoff_thresh)).fill(0);
  basis = orth(basis); // Not ortho at all; need to do explicitly
  assert((K+1) == basis.n_cols);
  return basis;
}

sp_mat make_radial_fourier_basis(const Points & points,
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
  basis = orth(basis); // Explicitly orthonormalize
  return sp_mat(basis);
}

sp_mat make_fourier_basis(const Points & points,
                        uint K,double max_freq){
  uint N = points.n_rows;
  mat basis = mat(N,K);

  basis.col(0) = ones<vec>(N);
  basis.col(1) = sin(datum::pi*sum(points,1));
  basis.col(2) = sin(datum::pi*(points.col(0) - points.col(1)));
  for(uint i = 3; i < K; i++){
    vec freq = 2.0*datum::pi * randi<vec>(2, distr_param(1,(uint)max_freq));
    vec flip = randn<vec>(1);
    if(flip(0) > 0.5)
      basis.col(i) = sin(points * freq);
    else
      basis.col(i) = cos(points * freq);
  }

  basis = orth(basis); // Explicitly orthonormalize
  return sp_mat(basis);
}

sp_mat make_voronoi_basis(const Points & points,
                       const Points & centers){
  VoronoiBasis v_basis(points,centers);
  return v_basis.get_basis();
}


VoronoiBasis::VoronoiBasis(const Points & points): m_points(points){
  n_basis = 0;
  n_dim = points.n_cols;
  n_points = points.n_rows;
};
VoronoiBasis::VoronoiBasis(const Points & points,
                           const Points & centers):
  m_points(points),m_centers(centers){
  assert(points.n_cols == centers.n_cols);
  m_dist = dist_mat(m_points,m_centers);
  n_basis = centers.n_rows;
  n_dim = points.n_cols;
  n_points = points.n_rows;
  assert(n_dim == centers.n_cols);
}

void VoronoiBasis::add_center(const vec & center){
  n_basis++;
  m_dist.resize(n_points,n_basis);
  m_centers.resize(n_basis,n_dim);
  replace_last_center(center);
}

void VoronoiBasis::replace_last_center(const vec & center){
  m_dist.col(n_basis-1) = lp_norm(m_points.each_row() - center.t(),2,1);
  m_centers.row(n_basis-1) = center.t();
}

uint VoronoiBasis::count(uint k) const{
  assert(k < n_basis);
  uvec P = col_argmin(m_dist); // Partition assignment
  return find(P == k).eval().n_elem;
}

uint VoronoiBasis::count_last() const{
  return count(n_basis - 1);
}

uint VoronoiBasis::min_count() const{
  uint mc = INT_MAX;
  for(uint k = 0; k < n_basis; k++){
    mc = min(mc,count(k));
  }
  return mc;
}

sp_mat VoronoiBasis::get_basis() const{
  umat loc = umat(2,n_points);
  uvec P = col_argmin(m_dist); // Partition assignment
  loc.row(0) = regspace<urowvec>(0,n_points-1).eval();
  loc.row(1) = P.t();
  vec data = ones(n_points);
  sp_mat basis = sp_mat(loc,data);
  return sp_normalise(basis,2,0); // l2 normalize each column
}


sp_mat make_basis(const string & mode,
                  const vector<double> & params,
                  const Points & points,
                  uint k){
  uint N = points.n_rows;
  sp_mat basis;  
  if(0 == mode.compare("voronoi")){
    // VORONOI
    assert(0 == params.size());
    Points centers = 2 * randu(k,2) - 1;
    VoronoiBasis vb = VoronoiBasis(points,centers);
    basis = vb.get_basis();
  }
  else if (0 == mode.compare("sample")){
    assert(0 == params.size());
    basis = make_sample_basis(N,k);
  }
  else if (0 == mode.compare("balls")){
    assert(1 == params.size());
    uint radius = (uint) params[0];
    Points centers = 2 * randu(k,2) - 1;
    basis = make_ball_basis(points,centers,radius);
  }
  else if (0 == mode.compare("rbf")){
    assert(1 == params.size());
    double bandwidth = (double) params[0];
    Points centers = 2 * randu(k,2) - 1;
    basis = make_rbf_basis(points,centers,bandwidth);
  }
  else{
    assert(false);
  }
  return basis;
}

LCP smooth_lcp(const sp_mat & smoother,
               const vector<sp_mat> & blocks,
               const mat & Q,
               const bvec & free_vars){
  uint n = smoother.n_rows;
  assert(n == smoother.n_cols);
  uint A = blocks.size();
  uint N = n*(A+1);
  assert(A >= 1);
  assert(size(n,n) == size(blocks.at(0)));
  assert(size(n,A+1) == size(Q));
  assert(N == free_vars.n_elem);
  
  // Smooth blocks
  vector<sp_mat> sblocks = block_rmult(smoother,blocks);

  // Smooth Q
  mat sQ = mat(size(Q));
  sQ.col(0) = Q.col(0); // State weights unchanged
  sQ.tail_cols(A) = smoother * Q.tail_cols(A);

  sp_mat M = build_M(sblocks);
  vec q = vectorise(sQ);
  return LCP(M,q,free_vars);
}

vector<sp_mat> make_freebie_flow_bases_ignore_q(const sp_mat & value_basis,
                                                const vector<sp_mat> blocks){
  vector<sp_mat> flow_bases;
  uint A = blocks.size();
  for(uint a = 0; a < A; a++){
    sp_mat raw_basis = blocks.at(a).t() * value_basis;
    flow_bases.push_back(sp_mat(orth(mat(raw_basis))));
    // Orthonorm (TODO: do directly in sparse?)
  }
  return flow_bases;
}

vector<sp_mat> make_freebie_flow_bases(const sp_mat & value_basis,
                                       const vector<sp_mat> blocks,
                                       const mat & Q){
  vector<sp_mat> flow_bases;
  uint A = blocks.size();
  assert((A+1) == Q.n_cols);
  for(uint a = 0; a < A; a++){
    mat raw_basis = join_horiz(mat(blocks.at(a).t() * value_basis),
                               Q.col(a+1));
    flow_bases.push_back(sp_mat(orth(raw_basis)));
    // Orthonorm (TODO: do directly in sparse?)
  }
  return flow_bases;
}


PLCP approx_lcp(const sp_mat & value_basis,
                const sp_mat & smoother,
                const vector<sp_mat> & blocks,
                const mat & Q,
                const bvec & free_vars){

  //Sizing and checking
  uint n = smoother.n_rows;
  assert(n == smoother.n_cols);
  uint A = blocks.size();
  assert(A >= 1);
  assert(size(n,n) == size(blocks.at(0)));
  assert(size(n,A+1) == size(Q));
  uint N = n*(A+1);
  assert(N == free_vars.n_elem);
  assert(n == value_basis.n_rows);

  // Smooth blocks
  vector<sp_mat> sblocks = block_rmult(smoother,blocks);

  // Build freebie flow bases for the smoothed problem
  bool ignore_q = false;
  vector<sp_mat> flow_bases;
  vec q;
  if(ignore_q){
    flow_bases = make_freebie_flow_bases_ignore_q(value_basis,
                                                  sblocks);
    // Project smoothed costs onto `freebie' basis
    mat sQ = mat(size(Q));  
    sQ.col(0) = Q.col(0);
    for(uint a = 0; a < A; a++){
      sp_mat F = flow_bases.at(a);
      sQ.col(a+1) = F * F.t() * smoother * Q.col(a+1);
    }
    q = vectorise(sQ);
  }
  else{
    mat sQ = mat(size(Q));  
    sQ.col(0) = Q.col(0);
    for(uint a = 0; a < A; a++){
      sQ.col(a+1) = smoother * Q.col(a+1);
    }
    q = vectorise(sQ);

    flow_bases = make_freebie_flow_bases(value_basis,
                                         sblocks,
                                         sQ);
  }
  // Build the basis blocks and the basis matrix
  block_sp_vec p_blocks;
  p_blocks.reserve(A + 1);
  p_blocks.push_back(value_basis);
  p_blocks.insert(p_blocks.end(),
                  flow_bases.begin(),
                  flow_bases.end());
  assert((A+1) == p_blocks.size());
  sp_mat P = block_diag(p_blocks);

  // Build LCP matrix M and the U coefficient matrix
  sp_mat M = build_M(sblocks);// + 1e-10 * speye(N,N); // Regularize
  sp_mat U = P.t() * M * P * P.t();
  return PLCP(P,U,q,free_vars);

}

