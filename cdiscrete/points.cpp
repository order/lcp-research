#include "misc.h"
#include "points.h"

#include <assert.h>
#include <set>

using namespace arma;
using namespace std;

uvec get_spatial_rows(const Points & points){
  return find_finite(points.col(0));
}

uvec get_special_rows(const Points & points){
  return find_nonfinite(points.col(0));
}

bool check_points(const Points & points){  
  if(points.is_empty()) return true;
  assert(!points.has_inf());
  
  uvec spatial = get_spatial_rows(points);
  assert(is_finite(points.rows(spatial)));

  uvec special = get_special_rows(points);
  uvec finite_idx = find_finite(points.rows(special));
  assert(finite_idx.is_empty());
  return true;
}


bool check_bbox(const mat & bbox){
  assert(2 == bbox.n_cols);
  assert(all(bbox.col(0) <= bbox.col(1))); // Must be full dimensional
  return true;
}


bool check_points_in_bbox(const Points & points, const mat & bbox){
  assert(check_bbox(bbox));
  if(points.is_empty()) return true;
  assert(check_points(points));

  uint D = points.n_cols;
  assert(D == bbox.n_rows);

  uvec spatial = get_spatial_rows(points);
  assert(is_finite(points.rows(spatial)));  // All finite
  for(uint d = 0; d < D; d++){
    uvec idx = uvec{d};
    assert(all(all(points(spatial,idx) >= bbox(d,0))));
    assert(all(all(points(spatial,idx) <= bbox(d,1))));
  }
  return true;
}


bool check_points_in_bbox(const TypedPoints & points, const mat & bbox){
  if(0 == points.num_spatial_nodes()) return true;
  
  uvec mask = points.get_spatial_mask();
  Points spatial_points = points.m_points.rows(mask);
  check_points_in_bbox(spatial_points, bbox);
}


/*
 * TYPED POINTS STRUCTURE
 */

TypedPoints::TypedPoints(const TypedPoints & points):
  m_reg(points.m_reg){
  m_points = points.m_points;
  n_rows = points.n_rows;
  n_cols = points.n_cols;
  _ensure_blanked();
}

TypedPoints::TypedPoints(const mat & points, const TypeRegistry & reg) :
  m_points(points), m_reg(reg){
  assert(m_points.n_rows >= m_reg.size());
  n_rows = points.n_rows;
  n_cols = points.n_cols;
  _ensure_blanked();
}

TypedPoints::TypedPoints(const mat & points) :
  m_points(points){
  n_rows = points.n_rows;
  n_cols = points.n_cols;
}

TypedPoints::TypedPoints(){}


void TypedPoints::register_type(uint idx, uint node_type){
  assert(idx <= m_points.n_rows);
  
  m_reg[idx] = node_type;
  m_points.row(idx).fill(SPECIAL_FILL);  // Blank out row
}

uint TypedPoints::num_special_nodes() const{
  return m_reg.size();
}

uint TypedPoints::num_all_nodes() const{
  return m_points.n_rows;
}


uint TypedPoints::num_spatial_nodes() const{
  return num_all_nodes() - num_special_nodes();
}

uvec TypedPoints::get_spatial_mask() const{
  return get_spatial_rows(m_points);
}

uvec TypedPoints::get_special_mask() const{
  return get_special_rows(m_points);
}

bool TypedPoints::is_special(uint idx) const{
  return m_reg.end() != m_reg.find(idx);
}

void TypedPoints::apply_typing_rule(const TypeRule & rule){
  /*
    Apply a single rule
   */
  TypeRegistry new_types = rule.type_elements(m_points);
  for(auto const & it : new_types){
    assert(m_reg.end() == m_reg.find(it.first)); // Not already in.
    assert(!m_points.row(it.first).has_nan()); // Not NaN filled.
    
    m_reg[it.first] = it.second;
    m_points.row(it.first).fill(datum::nan);
  }
  assert(check_validity());
}

void TypedPoints::apply_typing_rules(const TypeRuleList & rules){
  /*
    Iterate through list and apply node type rules
  */
  for(auto const & it : rules){
    apply_typing_rule(*it);
  }

}

void TypedPoints::apply_remapper(const NodeRemapper & remapper){
  remapper.remap(m_points);
  assert(check_validity());
}

void TypedPoints::apply_remappers(const NodeRemapperList & remappers){
  /*
    Iterate through list and apply node remappers
  */
  for(auto const & it : remappers){
    apply_remapper(*it);
  }
}

bool TypedPoints::check_validity() const{
  uint N = m_points.n_rows;
  for(uint i = 0; i < N; ++i){
    bool has_nan = m_points.row(i).has_nan();
    bool is_special = (m_reg.find(i) != m_reg.end());
    assert(has_nan == is_special);

    if(is_special){
      assert(m_reg.at(i) > SPATIAL_TYPE); // Special types not Euclidean
    }
  }

  for(auto const & it : m_reg){
    assert(it.first < N); // Valid index
    assert(it.second > SPATIAL_TYPE); // Special nodes only.
  }
  return true;
}

bool TypedPoints::check_in_bbox(const mat & bbox) const{
  return check_points_in_bbox(m_points, bbox);
}

bool TypedPoints::check_in_bbox(const arma::vec & low, const arma::vec & high) const{
  uint N = low.n_elem;
  assert(N == high.n_elem);
  mat bbox = mat(N,2);
  bbox.col(0) = low;
  bbox.col(1) = high;
  return check_in_bbox(bbox);
}


bool TypedPoints::equals(const TypedPoints & other) const{
  // Check dimensions
  if(other.n_rows != this->n_rows) return false;
  if(other.n_cols != this->n_cols) return false;
  if(other.m_reg.size() != this->m_reg.size()) return false;
  if(any(get_spatial_mask() != other.get_spatial_mask())) return false;
  if(any(get_special_mask() != other.get_special_mask())) return false;

  // Check the registry
  for(auto const& it : other.m_reg){
    uint idx = it.first;
    uint val = it.second;
    if(this->m_reg.end() == this->m_reg.find(idx)) return false;
    if(val != this->m_reg.at(idx)) return false;
  }
  
  // Check the coords
  
  if(0 == num_spatial_nodes()) return true;
  uvec sp_idx = get_spatial_mask();
  return approx_equal(other.m_points.rows(sp_idx),
		      this->m_points.rows(sp_idx),
		      "both", PRETTY_SMALL, PRETTY_SMALL);
}



ostream& operator<<(ostream& os, const TypedPoints& p){
  for(uint i = 0; i < p.n_rows; i++){
    os << "Point [" << i << "]:";
    if(p.is_special(i)){
      os << "\tSpecial (" << p.m_reg.find(i)->second << ")" << endl;
    }
    else{
      os << p.m_points.row(i);
    }
  }
  return os;
}

void TypedPoints::_ensure_blanked(){
  /*
    Make sure that everthing in the registry corresponds to NaN'd rows
  */
  for(auto const & it : m_reg){
    assert(!m_points.row(it.first).has_inf());
    bool is_blank = m_points.row(it.first).has_nan();
    if(!is_blank){
      m_points.row(it.first).fill(SPECIAL_FILL);
    }
    assert(0 == uvec(find_finite(m_points.row(it.first))).n_elem);
  }
}

/*
 * OUT OF BOUNDS NODE TYPE RULE
 */

OutOfBoundsRule::OutOfBoundsRule(const mat & bounding_box,
				 uint oob_type) :
  m_bbox(bounding_box), m_type(oob_type) {
  assert(check_bbox(m_bbox));
  assert(oob_type > TypedPoints::SPATIAL_TYPE);
}

TypeRegistry OutOfBoundsRule::type_elements(const mat & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(D == m_bbox.n_rows);
  assert(2 == m_bbox.n_cols);

  vec lb = m_bbox.col(0);
  vec ub = m_bbox.col(1);

  set<uint> violations;
  uvec spatial_rows = get_spatial_rows(points);
  // Iterate through the dimensions
  for(uint d = 0; d < D; ++d){
    uvec dim_col = uvec{d};
    uvec low_mask = find(points(spatial_rows,dim_col) < lb(d));
    low_mask = spatial_rows(low_mask);
    violations.insert(low_mask.begin(), low_mask.end());
    
    uvec high_mask = find(points(spatial_rows,dim_col) > ub(d));
    high_mask = spatial_rows(high_mask);
    violations.insert(high_mask.begin(), high_mask.end());
  }

  TypeRegistry ret;
  for(auto const & it : violations){
    assert(m_type > TypedPoints::SPATIAL_TYPE);
    ret[it] = m_type;
  }
  return ret;
}


/*
 * SATURATION REMAPPER
 */

SaturateRemapper::SaturateRemapper(const mat & bounding_box) :
  m_bbox(bounding_box){
  assert(check_bbox(m_bbox));
}


void SaturateRemapper::remap(Points & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(D == m_bbox.n_rows);
  assert(2 == m_bbox.n_cols);

  uvec spatial_rows = get_spatial_rows(points);
  for(uint d = 0; d < D; ++d){
    uvec dim_col = uvec{d};
    double lb = m_bbox(d,0);
    double ub = m_bbox(d,1);
    if(lb == -datum::inf && ub == datum::inf){
      continue;
    }
    
    uvec low_mask = find(points(spatial_rows,dim_col) < lb);
    low_mask = spatial_rows(low_mask);
    points(low_mask,dim_col).fill(lb);

    uvec high_mask = find(points(spatial_rows,dim_col) > ub);
    high_mask = spatial_rows(high_mask);
    points(high_mask,dim_col).fill(ub);
  }
  assert(check_points_in_bbox(points,m_bbox));
}

void SaturateRemapper::remap(TypedPoints & points) const{
  remap(points.m_points);
}


/*
 * WRAP REMAPPER
 */

WrapRemapper::WrapRemapper(const mat & bounding_box) :
  m_bbox(bounding_box){
  assert(check_bbox(m_bbox));
}


void WrapRemapper::remap(Points & points) const{
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(D == m_bbox.n_rows);

  uvec spatial_rows = get_spatial_rows(points);
  for(uint d = 0; d < D; ++d){
    uvec dim_col = uvec{d};
    double lb = m_bbox(d,0);
    double ub = m_bbox(d,1);   

    if(lb == -datum::inf || ub == datum::inf){
      assert(lb == -datum::inf && ub == datum::inf);
      continue;
    }
    
    assert(is_finite(m_bbox.row(d)));

    vec tmp = points(spatial_rows,dim_col) - lb;
    tmp = vec_mod(tmp, ub - lb);
    points(spatial_rows,dim_col) = tmp + lb;
    
    assert(not any(points.col(d) > ub));
    assert(not any(points.col(d) < lb));
  }
  assert(check_points_in_bbox(points,m_bbox));
}

void WrapRemapper::remap(TypedPoints & points) const{
  remap(points.m_points);
}
