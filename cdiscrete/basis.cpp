#include <assert.h>
#include "basis.h"
#include "grid.h"
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
  
  vec gauss = exp(-bandwidth * sum(pow(points.each_row() - center.t(),2),1));
  return gauss;
}

vec gaussian(const Points & points,
             const vec & center,
             const mat & cov){
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(D == center.n_elem);
  assert(size(D,D) == size(cov));
  
  vec gauss = vec(N);
  for(uint i = 0; i < N; i++){
    vec x = points.row(i).t() - center;
    gauss(i) = exp(- dot(x, cov * x));
  }
  return gauss;
}
vec laplacian(const Points & points,
	      const vec & center,
	      const double bandwidth){
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(D == center.n_elem);
  
  vec laplace = exp(-bandwidth *
		    sqrt(sum(pow(points.each_row() - center.t(),2),1)));
  return laplace;
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

vec make_ball(const Points & points,
	      const vec & center,
	      const double radius){
  uint N = points.n_rows;
  vec ball = zeros<vec>(N);
  vec dist = lp_norm(points.each_row() - center.t(),2,1);
  ball(find(dist <= radius)).fill(1);
  return ball;
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


mat make_rbf_basis(const TypedPoints & points,
                   const Points & centers,
                   double bandwidth,
                   double cutoff_thresh){
  uint N = points.n_rows;
  uint K = centers.n_rows;
  uint S = points.num_special_nodes();
  
  mat basis = zeros<mat>(N,K+1+S);
  for(uint k = 0; k < K; k++){
    basis.col(k) = gaussian(points.m_points, centers.row(k).t(), bandwidth);
  }

  basis.col(K) = ones<vec>(N); // All ones

  if(S > 0){
    uvec special_rows = points.get_special_mask();
    uvec special_cols = regspace<uvec>(K + 1, K + S); // regspace include end
    assert(S == special_rows.n_elem);
    assert(S == special_cols.n_elem);
    basis.rows(special_rows).fill(0); // Zero out special
    basis.submat(special_rows, special_cols) = eye(S,S);
  }
  
  basis(find(basis < cutoff_thresh)).fill(0);
  basis = orth(basis); // Not ortho at all; need to do explicitly
  if(basis.n_cols < (K+1+S)){
    cerr << "WARNING: Basis degenerate... ("
	 << basis.n_cols << "/" << (K + S + 1) << " cols )"  << endl;
  }
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
  
  basis(find(basis < cutoff_thresh)).fill(0);
  basis = orth(basis); // Not ortho at all; need to do explicitly
  if(basis.n_cols < (K+1)){
    cerr << "WARNING: Basis degenerate..." << endl;
  }
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
  VoronoiBasis v_basis(points, centers);
  return v_basis.get_basis();
}

sp_mat make_voronoi_basis(const TypedPoints & points,
                       const Points & centers){
  TypedVoronoiBasis v_basis(points, centers);
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
  m_dist = dist_mat(m_points, m_centers);
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

////////////////////////////////////////////////////////////////////////////
// Typed Voronoi Basis

TypedVoronoiBasis::TypedVoronoiBasis(const TypedPoints & points,
				     const Points & centers) : 
  m_typed_points(points), m_centers(centers)
{
  assert(points.n_cols == centers.n_cols);
  m_dist = dist_mat(m_typed_points.m_points,m_centers);
  n_basis = centers.n_rows;
  n_dim = points.n_cols;
  n_points = points.n_rows;
  assert(n_dim == centers.n_cols);
  assert(m_dist.n_rows == points.n_rows);
  assert(m_dist.n_cols == centers.n_rows);
}


sp_mat TypedVoronoiBasis::get_basis() const{
  umat loc = umat(2,n_points);
  uvec P = uvec(n_points);
  uvec spatial_mask = m_typed_points.get_spatial_mask();
  P.rows(spatial_mask)= col_argmin(m_dist.rows(spatial_mask));

  // Only supporting one OOB for now.
  uint oob_idx = m_centers.n_rows;
  assert(1 == m_typed_points.num_special_nodes());
  P(m_typed_points.get_special_mask()).fill(oob_idx);
  assert(max(P) == oob_idx);

  // Convert into a space matrix
  loc.row(0) = regspace<urowvec>(0,n_points-1).eval();
  loc.row(1) = P.t();
  vec data = ones(n_points);
  sp_mat basis = sp_mat(loc,data);
  return sp_normalise(basis,2,0); // l2 normalize each column
}

////////////////////////////////////////////////////////////////////////////
// Freebie basis functions //
/////////////////////////////

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

sp_mat make_freebie_flow_basis(const sp_mat value_basis, const sp_mat block){
  return block.t() * value_basis;
}

sp_mat make_freebie_value_basis(const sp_mat flow_basis, const sp_mat block){

  sp_mat value_basis = sp_mat(size(flow_basis));
  uint K = flow_basis.n_cols;
  for(uint i = 0; i < K; i++){
    // Iterate throught columns and do a sparse matrix solve
    value_basis.col(i) = spsolve(block.t(),
				 vec(flow_basis.col(i)));
  }
  return value_basis;
}


vector<sp_mat> balance_bases(const vector<sp_mat> initial_bases,
			      const vector<sp_mat> blocks){
  /*
   * Take in initial bases, and balanced them via the S&S "freebie" 
   * relationship
   */
  
  uint A = blocks.size();
  assert(A + 1 == initial_bases.size());
  assert(A > 0);


  // Explicitly add initial value basis
  vector<sp_mat> value_basis_vector;
  value_basis_vector.push_back(initial_bases.at(0));
  
  // Add freebie flow components so the the value basis will span the initial
  // flow basis
  for(uint i = 0; i < A; i++){
    if(0 == initial_bases.at(i+1).n_cols) continue;
    sp_mat val_comp = make_freebie_value_basis(initial_bases.at(i+1),
					       blocks.at(i));
    value_basis_vector.push_back(val_comp);
  }

  // Join value components into a single sparse matrix
  sp_mat balanced_value_basis = h_join_sp_mat_vector(value_basis_vector);

  // Generated the freebie flow bases; each should span the initial flow bases
  vector<sp_mat> balanced_bases_vector;
   balanced_bases_vector.push_back(balanced_value_basis);
  for(uint i = 0; i < A; i++){
    sp_mat balanced_flow_basis = make_freebie_flow_basis(balanced_value_basis,
					       blocks.at(i));
    balanced_bases_vector.push_back(balanced_flow_basis);
  }

  return balanced_bases_vector;
}


vector<sp_mat> make_freebie_flow_bases_ignore_q(const sp_mat & value_basis,
                                                const vector<sp_mat> blocks){
  vector<sp_mat> flow_bases;
  uint A = blocks.size();
  for(uint a = 0; a < A; a++){
    cout << "\t\tMaking freebie flow basis " << a << " (ignoring q)..."
	 << endl;
    sp_mat raw_basis = blocks.at(a).t() * value_basis;
    flow_bases.push_back(sp_mat(orth(mat(raw_basis))));
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
     cout << "\t\tMaking freebie flow basis " << a << "..."
	  << endl;
    mat raw_basis = join_horiz(mat(blocks.at(a).t() * value_basis),
                               Q.col(a+1));
    flow_bases.push_back(sp_mat(orth(raw_basis)));
    // Orthonorm (TODO: do directly in sparse?)
  }
  return flow_bases;
}
vector<mat> make_raw_freebie_flow_bases(const mat & value_basis,
                                        const vector<sp_mat> blocks,
                                        const mat & Q){
  // Same as above, but don't orthonormalize
  vector<mat> flow_bases;
  uint A = blocks.size();
  assert((A+1) == Q.n_cols);
  for(uint a = 0; a < A; a++){
    mat raw_basis = join_horiz(mat(blocks.at(a).t() * value_basis),
                               Q.col(a+1));
    flow_bases.push_back(raw_basis);
  }
  return flow_bases;
}

PLCP approx_lcp(const sp_mat & value_basis,
                const sp_mat & smoother,
                const vector<sp_mat> & blocks,
                const mat & Q,
                const bvec & free_vars,
		bool ignore_q){

  cout << "Approximating LCP as PLCP..." << endl;

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
  cout << "\tSmoothing transition blocks..." << endl;
  vector<sp_mat> sblocks = block_rmult(smoother, blocks);

  // Build freebie flow bases for the smoothed problem
  vector<sp_mat> flow_bases;
  vec q;
  cout << "\tForming freebie bases..." << endl;
  if(ignore_q){
    cout << "\t\tIgnoring Q" << endl; 
    flow_bases = make_freebie_flow_bases_ignore_q(value_basis,
                                                  sblocks);
    // Project smoothed costs onto `freebie' basis
    cout << "\t\tProjecting Q onto bases..." << endl;
    mat sQ = mat(size(Q));  
    sQ.col(0) = Q.col(0);
    for(uint a = 0; a < A; a++){
      cout << "\t\t\tAction " << a << "..." << endl;

      sp_mat F = flow_bases.at(a);
      sQ.col(a+1) = F * (F.t() * (smoother * Q.col(a+1)));
    }
    q = vectorise(sQ);
  }
  else{
    cout << "\t\tUsing Q" << endl; 

    mat sQ = mat(size(Q));  
    sQ.col(0) = Q.col(0);
    sQ.tail_cols(A) = smoother * Q.tail_cols(A);
    q = vectorise(sQ);

    flow_bases = make_freebie_flow_bases(value_basis,
                                         sblocks,
                                         sQ);
  }
  cout << "\tBuilding block diagonal Phi..." << endl;
  // Build the basis blocks and the basis matrix
  block_sp_vec p_blocks;
  p_blocks.reserve(A + 1);
  p_blocks.push_back(value_basis);
  p_blocks.insert(p_blocks.end(),
                  flow_bases.begin(),
                  flow_bases.end());
  assert((A+1) == p_blocks.size());
  sp_mat P = block_diag(p_blocks);

  cout << "\tForming coefficients U..." << endl;
  // Build LCP matrix M and the U coefficient matrix
  sp_mat M = build_M(sblocks);
  sp_mat U = P.t() * M; // * P * P.t();
  return PLCP(P,U,q,free_vars);

}


////////////////////////////////////////////////////////////////////////////
// Table-based variable resolution basis factory //
///////////////////////////////////////////////////

TabularVarResBasis::TabularVarResBasis(const TypedPoints & points, mat bounds) :
  m_typed_points(points)
{
  /*
   * This is the constructor for a variable resolution "grid" basis. E.g.
   * o--oo-o
   * |  || |
   * |  |o-o
   * |  |o-o
   * |  || |
   * o--oo-o
   * For the tabular basis, all values in the same cell have the same value.
   * Each "cell" is a rectangular set of points.
   */
  typedef vector<uint> stduvec;
  n_bases = 0;
  n_rows = points.n_rows;
  n_cols = points.n_cols;
  
  assert(size(n_cols, 2) == size(bounds));
  assert(1 >= points.num_special_nodes());

  // Add the special basis
  if(points.num_special_nodes() > 0){
    m_basis_bounds.push_back(mat(0,0));

    stduvec special_idx = conv_to<stduvec>::from(points.get_special_mask());
    m_point_assign.push_back(special_idx);
    
    n_bases++;
  }
  
  // Add the spatial basis
  m_basis_bounds.push_back(bounds);
  stduvec spatial_idx = conv_to<stduvec>::from(points.get_spatial_mask());
  m_point_assign.push_back(spatial_idx);
  
  n_bases++;
}


sp_mat TabularVarResBasis::get_basis() const{
  /*
   * Convert the internal representation into an Nxk sparse basis matrix
   */
  assert(n_bases > 0);

  umat loc = umat(2, n_rows);
  vec data = vec(n_rows);
  assert(m_point_assign.size() == n_bases);
  uint I = 0;
  for(uint i = 0; i < n_bases; i++){
    double basis_points = (double)m_point_assign.at(i).size();
    double p = 1.0 / basis_points;
    assert(basis_points > 0);
    for(auto const& it : m_point_assign.at(i)){
      loc(0,I) = it;
      loc(1,I) = i;
      data(I) = p;
      I++;
    }
  }
  return sp_mat(loc, data);
}


bool TabularVarResBasis::can_split(uint basis_idx, uint dim_idx) const{
  /*
   * Checks if the basis vector can be split on a dimension.
   * Makes sure that each resulting basis has at least one point
   * NB: currently assumes a regular spacing; will need to modify for
   * irregular points
   */
  assert(dim_idx < n_cols);
  assert(basis_idx < n_bases);

  vector<uint> std_indices =  m_point_assign.at(basis_idx);
  if(std_indices.size() < 2){
    return false; // Singleton basis
  }
  uvec indices = conv_to<uvec>::from(std_indices);

  // Check that there is diversity in the column
  vec split_col = m_typed_points.m_points.submat(indices, uvec{dim_idx});
  if(abs(min(split_col) - max(split_col)) < PRETTY_SMALL){
    return false;
  }

  return true;
}


uint TabularVarResBasis::split_basis(uint basis_idx, uint dim_idx){
  /*
   * Splits a basis cell along a given dimension.
   * Updates the old basis as the [lo,mid) cell, and makes a new cell
   * as the [mid, hi) cell.
   *
   * Returns the new basis' index
   */
  
  assert(can_split(basis_idx, dim_idx));
  mat bounds = m_basis_bounds[basis_idx];
  vector<uint>& assign = m_point_assign[basis_idx];

  double lo = bounds(dim_idx, 0);
  double hi = bounds(dim_idx, 1);
  double mid = 0.5 * (lo + hi);
  uint new_basis_idx = m_basis_bounds.size();
  n_bases++;

  // Make new basis
  m_basis_bounds.push_back(bounds);  // Copies
  m_basis_bounds[new_basis_idx](dim_idx,0) = mid;
  
  // Update old basis
  m_basis_bounds[basis_idx](dim_idx,1) = mid;


  // Split assignment
  vector<uint> old_basis;
  vector<uint> new_basis;
  for(auto const & it : assign){
    if(m_typed_points.m_points(it, dim_idx) < mid){
      old_basis.push_back(it);
    }
    else{
      new_basis.push_back(it);
    }
  }

  // Update old assignment
  m_point_assign[basis_idx] = old_basis;
  
  // Add new assignment
  m_point_assign.push_back(new_basis);

  // Return the new basis index
  return m_point_assign.size() - 1;
}

umat box2points(const umat & bbox, uvec grid_size){
  uint D = grid_size.n_elem;
  vector<uvec> marginal;
  for(uint d = 0; d < D; d++){
    marginal.push_back(regspace<uvec>(bbox(d,0), bbox(d,1)));
  }
  return make_points<umat, uvec>(marginal);
}

umat box2vert(const umat & bbox, uvec grid_size){
  uint D = grid_size.n_elem;
  vector<uvec> marginal;
  for(uint d = 0; d < D; d++){
    marginal.push_back(conv_to<uvec>::from(bbox.row(d)));
  }
  return make_points<umat,uvec>(marginal);
}

uvec box2width(const umat & bbox){
  /*
   * Number of points in the box. Note that [0,0] has one point.
   */
  uint D = bbox.n_rows;
  uvec width = uvec(D);
  for(uint d = 0; d < D; d++){
    assert(bbox(d,1) >= bbox(d,0));
    width(d) = bbox(d,1) - bbox(d,0) + 1;
  }
  return width;
}

////////////////////////////////////////////////////////////////////////////
// Multilinear interpolation based variable resolution basis factory //
///////////////////////////////////////////////////////////////////////

MultiLinearVarResBasis::MultiLinearVarResBasis(const uvec & grid_size)
  : m_grid_size(grid_size)
{
  /*
   * Takes in a uvec that discribes the number of points in the underlying
   * grid discretization.
   * 
   * This underlying grid defines the point index <-> coordinate mapping.
   * so if grid_size is [3,4], then this is a 2x3 set of 
   * underlying cells, with vertices indexed in C-order (row-major):
   * 
   * 3   9 -10 -11
   *     |   |   |
   * 2   6 - 7 - 8
   *     |   | x | <-- (0.5, 1.1) here
   * 1   3 - 4 - 5
   *     |   |   |
   * 0   0 - 1 - 2
   *
   *    -1   0   1
   * then continuous point (0.5,1.1) is in the underlying grid cell 
   * [0,1] x [1,2], which has discrete 
   * vertices coordinates: [1,1], [1,2], [2,1], [2,2]
   * These correspond to continuous vertices: 
   * (0, -2), (0,-1), (1, -2), (1,-1)
   * And indices: [4, 7, 5, 8]
   */
  
  uint n_dim = grid_size.size();

  // Sets up a single basis grid cell for the discretized space.
  umat box = zeros<umat>(n_dim,2);
  box.col(1) = (m_grid_size - 1);
  m_cell_to_bbox.push_back(box);

  m_num_oob = 1;
}

sp_mat MultiLinearVarResBasis::get_basis() const{
  /*
   * Create the basis from the multi linear grid.
   *
   * Each basis column represents is a cell vertex, and its support is the
   * set of elements that depend on that vertex.
   */
  uint n_weights = 0;
  uint D = m_grid_size.n_elem; // Number of dimensions
  uint V = pow(2,D); // Number of vertices per rectangle

  uint nnz = 0;
  // point_id -> {vertex_id -> weight}
  map<uint, map<uint, double>> weight_map;
  
  // Stores the points found as bases with non-zero weights.
  // There will be a bases around each.
  set<uint> bases;

  for(auto const & cell_box : m_cell_to_bbox){
    uvec width = box2width(cell_box);
    uint P = prod(width);

    // Set up the vertices and their indices
    umat vertices = box2vert(cell_box, m_grid_size);
    uvec vidx = coords_to_indices(m_grid_size,
				  Coords(conv_to<imat>::from(vertices)));
    assert(size(V,D) == size(vertices));

    // Set up the points and their indices
    umat points = box2points(cell_box, m_grid_size);
    assert(size(P,D) == size(points));
    uvec pidx = coords_to_indices(m_grid_size,
				  Coords(conv_to<imat>::from(points)));
    assert(P == pidx.n_elem);

    // Subtract out the least vertex
    mat delta = row_diff(conv_to<mat>::from(points),
			 conv_to<rowvec>::from(vertices.row(0)));
    assert(all(all(delta >= -ALMOST_ZERO)));

    // Divide by the width of the cell
    delta = row_divide(delta, conv_to<rowvec>::from(width.t() - 1));
    assert(all(all(delta >= -ALMOST_ZERO)));
    assert(all(all(delta <= 1 + ALMOST_ZERO)));
    scalar_max_inplace(delta, 0); 
    scalar_min_inplace(delta, 1); // Ensure [0,1] 
    // Now is the fraction of the cell away from the least vertex

    // Do the multilinear interpolation
    uint halfV = V / 2;
    mat weights = ones<mat>(P, V);  // Initialize with unity weights
    bvec mask;
    mat rep_delta;
    for(uint d = 0; d < D; d++){
      /*
	---===Interpolation code explaination===---
	For 2D grid on box [0,A] x [0,B] there are 4 vertices in C-order:
	     x y
	0 -> 0 0
	1 -> 0 B
	2 -> A 0
	3 -> A B
	NB: each vertex also has a point index; but these are there "local
	vertices indices" for the purposes of talking about weights.
	
	Consider the point (1, 1). It's multilinear interpolation weights
	over vertices is:
	0 -> [(A - 1) / A] * [(B - 1) / B]
	1 -> [(A - 1) / A] * [1 / B]
	2 -> [1 / A] * [(B - 1) / B]
	3 -> [1 / A] * [1 / B]
	Notice that the contribution of the x-dim distances are:
	 [(A - 1) / A]
	 [(A - 1) / A]
	 [1 / A]
	 [1 / A]
	 So the behavior of the first column's contribution to the weights
	 is the behavior of the most-significant bit in binary.

	 The delta matrix holds the [1 / A, 1 / B] values.
      */

      // This mask is the (D-d-1)^th column of the binary expansion matrix
      // [[0 ... 0], [0 ... 1], ..., [1 ... 1]
      mask = binmask(D-d-1, D);

      // If you match the 1 you get the (1 / A) contribution
      rep_delta = repmat(delta.col(d), 1, halfV); // Applies to exactly half
      weights.cols(find(mask == 1)) %= rep_delta;

      // If you match the 0 you get the (A - 1) / A contribution
      weights.cols(find(mask == 0)) %= 1 - rep_delta;
    }

    //Basic sanity check: weights seem reasonable..
    assert(is_finite(weights));
    assert(all(all(weights >= 0))); 
    assert(all(all(weights <= 1)));
    assert(abs(accu(weights) - P) < PRETTY_SMALL);

    // Convert the weight matrix into the weight map
    for(uint p = 0; p < P; p++){
      uint pid = pidx[p];
      if(weight_map.end() != weight_map.find(pid)){
	// Already represented (not in interior of a cell)
	continue;
      }
      for(uint v = 0; v < V; v++){
	uint vid = vidx[v];
	if(weights(p,v) < ALMOST_ZERO){
	  // Zero; ignore
	  continue;
	}
	bases.insert(vid);
	weight_map[pid][vid] = weights(p,v);
	nnz++;
      }
    }
  }

  // Index the vertices as bases
  map<uint, uint> vert_to_basis_map;
  uint basis_id = 0;
  for(auto const & it : bases){
    vert_to_basis_map[it] = basis_id++;
  }

  // Convert to sp_mat
  umat loc = umat(2,nnz + m_num_oob);
  vec data = vec(nnz + m_num_oob);
  uint I = 0;
  for(auto const & pit : weight_map){
    for(auto const & vit : pit.second){
      loc(0,I) = pit.first;
      loc(1,I) = vert_to_basis_map[vit.first];
      data(I) = vit.second;
      I++;
    }
  }

  // TODO: Sanity check (all points covered)
  uint n_nodes = prod(m_grid_size) + m_num_oob;
  uint n_bases = bases.size() + m_num_oob;
  
  // Add OOB as isolated node
  loc(0,nnz) = n_nodes - 1;
  loc(1,nnz) = n_bases - 1;
  data(nnz) = 1.0;

  return sp_mat(loc, data, n_nodes, n_bases);
}

bool MultiLinearVarResBasis::can_split(uint cell_idx, uint dim_idx) const{
  /*
   * Check that cell_idx can be split along dim_idx.
   *
   * Needs to be at least one intermediate points along that dimension.
   * E.g. if the interval is [1,2] then there are no intermendiate points
   * and we cannot split.
   */
  assert(cell_idx < m_cell_to_bbox.size());
  umat cell_box = m_cell_to_bbox[cell_idx];
  // Needs to be at least one intermediate points along that dimension
  // E.g. if the interval is [1,2] then there are no intermendiate points
  // And we cannot split.
  return cell_box(dim_idx, 1) > cell_box(dim_idx, 0) + 1;
}

uint MultiLinearVarResBasis::split_cell(uint cell_idx, uint dim_idx){
  assert(can_split(cell_idx, dim_idx));

  umat cell_bbox = m_cell_to_bbox[cell_idx];
  uint width = cell_bbox(dim_idx, 1) - cell_bbox(dim_idx, 0) + 1;
  assert(width >= 3);
  uint mid = width / 2; // floor

  // Add new bbox
  umat new_bbox = umat(cell_bbox);
  new_bbox(dim_idx,0) = mid;  // [mid, high]
  m_cell_to_bbox.push_back(new_bbox);

  // Update old bbox
  m_cell_to_bbox[cell_idx](dim_idx,1) = mid;  // [low, mid]

  return m_cell_to_bbox.size() - 1;
}

void MultiLinearVarResBasis::split_per_dimension(uint cell_idx, uvec split){
  /*
   * Split an indexed cell multiple times.
   * cell_idx: the cell index to split
   * split: split[d] is the number of times to split the cell in half along
   *        dimension d.
   *
   * Example: if cell_idx i is the set of points [0,2] x [0,2] then 
   *   split_per_dimension(i,[1,1]) will split this cell once per dimension,
   *   leading to 4 cells: [0,1]x[0,1], [0,1]x[1,2], [1,2]x[0,1], [1,2]x[1,2]
   *   Note that since none of these cells contain "intermediate nodes", 
   *   these cells will behave, locally, as the identity basis.
   * Example: if cell_idx i is [0,4]x[0,4], then split_per_dimension(i,[1,2])
   *   will create 8 cells:
   *   [0,2]x[0,1], [0,2]x[1,2],...,[2,4]x[0,1],...,[2,4]x[3,4]
   */

  // Make sure the input is sane.
  assert(cell_idx < m_cell_to_bbox.size()); // Valid cell id
  uint n_dim =  m_grid_size.n_elem;
  assert(n_dim == split.n_elem); // Same dimension
  
  // Split the cells.
  vector<uint> frontier;
  frontier.push_back(cell_idx);
  uint n_new_bases = prod(split + 1);
  frontier.reserve(n_new_bases);
  for(uint split_dim = 0; split_dim < n_dim; split_dim++){
    for(uint i = 0; i < split[split_dim]; i++){
      vector<uint> new_elements;
      for(auto const & it : frontier){
	assert(can_split(it, split_dim));
	uint new_cell_id = split_cell(it, split_dim);
	new_elements.push_back(new_cell_id);
      }
      frontier.insert(frontier.end(),
		      new_elements.begin(),
		      new_elements.end());
    }
  }
  assert(n_new_bases == frontier.size());
}
