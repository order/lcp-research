#include "misc.h"
#include "points.h"

#include <assert.h>
#include <set>

using namespace arma;
using namespace std;

uvec get_spatial_rows(const Points & points){
  assert(check_points(points));
  return find_finite(points.col(0));
}

uvec get_special_rows(const Points & points){
  assert(check_points(points));
  return find_nonfinite(points.col(0));
}

bool check_points(const Points & points){
  assert(!points.has_inf());
  
  // No non-finite elements in the "spatial" rows
  uvec spatial = find_finite(points.col(0));
  assert(is_finite(points.rows(spatial)));

  // No finite elements in the "special" rows
  uvec special = find_nonfinite(points.col(0));
  assert(0 == uvec(find_finite(points.rows(special))).n_elem);
  return true;
}


bool check_bbox(const mat & bbox){
  assert(2 == bbox.n_cols);
  assert(all(bbox.col(0) < bbox.col(1))); // Must be full dimensional
  return true;
}


bool check_points_in_bbox(const Points & points, const mat & bbox){
  assert(check_bbox(bbox));
  uint D = points.n_cols;
  assert(D == bbox.n_rows);
  
  for(uint d = 0; d < D; d++){
    assert(not any(points.col(d) < bbox(d,0)));
    assert(not any(points.col(d) > bbox(d,1)));
  }
  return true;
}


/*
 * TYPED POINTS STRUCTURE
 */

TypedPoints::TypedPoints(const mat & points, const TypeRegistry & reg) :
  m_points(points), m_reg(reg){
  assert(m_points.n_rows >= m_reg.size());
  _ensure_blanked();
}

TypedPoints::TypedPoints(const mat & points) :
  m_points(points){}

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
}

void TypedPoints::apply_typing_rules(const TypeRuleList & rules){
  /*
    Iterate through list and apply node type rules
  */
  for(auto const & it : rules){
    apply_typing_rule(it);
  }
}

void TypedPoints::apply_remapper(const NodeRemapper & remapper){
  remapper.remap(m_points);
}

void TypedPoints::apply_remappers(const NodeRemapperList & remappers){
  /*
    Iterate through list and apply node remappers
  */
  for(auto const & it : remappers){
    apply_remapper(it);
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
    uvec low_mask = find(points(spatial_rows,dim_col) < lb);
    low_mask = spatial_rows(low_mask);
    violations.insert(low_mask.begin(), low_mask.end());
    
    uvec high_mask = find(points(spatial_rows,dim_col) > ub);
    high_mask = spatial_rows(high_mask);
    violations.insert(high_mask.begin(), high_mask.end());
  }

  TypeRegistry ret;
  for(auto const & it : violations){
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
    assert(is_finite(m_bbox.col(d)));

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
