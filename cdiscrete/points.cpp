#include "points.h"

using arma;
using std;

/*
 * TYPED POINTS STRUCTURE
 */

TypedPoints::TypedPoints(const mat & points, const NodeTypeRegistry & reg) :
  m_points(points), m_reg(reg), m_max_type(EUCLIDEAN_TYPE) {
  assert(m_points.n_rows >= m_reg.size());
  _ensure_blanked();
}

TypedPoints::TypedPoints(const mat & points) :
  m_points(points), m_max_type(EUCLIDEAN_TYPE) {}

TypedPoints::TypedPoints() : m_max_type(EUCLIDEAN_TYPE){}

uint TypedPoints::get_next_type(){
  return ++m_max_type;
}

void TypedPoints::register(Index idx, NodeType node_type){
  assert(node_type <= m_max_type);
  assert(idx <= m_points.n_rows);
  
  m_reg[idx] = node_type;
  m_points.row(idx).fill(SPECIAL_FILL);  // Blank out row
}

uint TypedPoints::num_special_nodes() const{
  return m_reg.size();
}

uint TypedPoints::num_normal_nodes() const{
  assert(m_points.n_rows >= m_reg.size());  
  return m_points.n_rows - m_reg.size();
}

uint TypedPoints::num_all_nodes() const{
  return m_points.n_rows();
}

uvec TypedPoints::get_normal_mask() const{
  uvec mask = ones(this->num_all_nodes());
  for(NodeTypeRegistry::const_iterator it = m_reg.begin();
      it != m_reg.end(); ++it){
    mask[it->first] = 0;
  }
  assert(this->num_normal_nodes() == sum(mask));
  return mask;
}

uvec TypedPoints::get_special_mask() const{
  uvec mask = zeros(this->num_all_nodes());
  for(NodeTypeRegistry::const_iterator it = m_reg.begin();
      it != m_reg.end(); ++it){
    mask[it->first] = 1;
  }
  assert(this->num_special_nodes() == sum(mask));
  return mask;
}

void TypedPoints::apply_typing_rule(const NodeTypeRule & rule){
  /*
    Apply a single rule
   */
  NodeTypeRegistry new_types = rule.type_elements(m_points);
  for(NodeTypeRegistry reg_it = new_types.begin();
      reg_it != new_types.end(); ++reg_it){
    Index row_idx = reg_it->first;
    NodeType row_type = reg_it->second;
    assert(!m_points.row(row_idx).has_nan());  // Check NaN <=> typed

    // Add to the registry and NaN fill the row
    m_reg[row_idx] = row_type;
    m_points.row(row_idx).fill(datum::nan);
  }
}

void TypedPoints::apply_typing_rules(const NodeTypeRuleList & rules){
  /*
    Iterate through list and apply node type rules
  */
  auto bound_fn = bind(&TypedPoints::apply_typing_rule,
		       this,
		       std::placeholders::_1);
  for_each(rules.begin(), rules.end(), bound_fn);
}

void TypedPoints::apply_remapper(const NodeRemapper & remapper){
  remapper->remap(m_points);
}

void TypedPoints::apply_remappers(const NodeRemapperList & remappers){
  /*
    Iterate through list and apply node remappers
  */
  auto bound_fn = bind(&TypedPoints::apply_remapper,
		       this,
		       std::placeholders::_1);
  for_each(rules.begin(), rules.end(), bound_fn);
}

bool TypedPoints::check_validity() const{
  uint N = m_points.n_rows;
  for(uint i = 0; i < N; ++i){
    bool has_nan = m_points.row(i).has_nan();
    bool is_special = (m_reg.find(i) != m_reg.end());
    assert(has_nan == is_special);

    if(is_special){
      assert(m_reg[i] > EUCLIDEAN_TYPE); // Special types not Euclidean
      assert(m_reg[i] <= m_max_type);
    }
  }
}

void TypedPoints::_ensure_blanked(){
  /*
    Make sure that everthing in the registry corresponds to NaN'd rows
  */
  for(NodeTypeRegistry::const_iterator it = m_reg.begin();
      it != m_reg.end(); ++it){
    bool is_blank = m_points.row(it->first).has_nan();
    if(!is_blank){
      m_points.row(it->first).fill(datum::nan);
    }
  }
}


/*
 * OUT OF BOUNDS NODE TYPE RULE
 */

OutOfBoundsRule::OutOfBoundsRule(const mat & bounding_box,
				 NodeType oob_type) :
  m_bbox(bounding_box), m_type(oob_type) {
  assert(check_bounding_box(m_bbox));
}

NodeTypeRegistry OutOfBoundsRule::type_elements(const TypedPoints & points){
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(D == m_bbox.n_rows);
  assert(2 == m_bbox.n_cols);

  vec lb = m_bbox.col(0);
  vec ub = m_bbox.col(1);

  set violations;

  // Iterate through the dimensions
  for(uint d = 0; d < D; ++d){
    uvec low = find(points.col(d) < lb(d));
    violations.insert(low.begin(), low.end());
    
    uvec high = find(points.col(d) > ub(d));
    violations.insert(high.begin(), high.end());
  }

  NodeTypeRegistry ret;
  for(set::const_iterator it = violations.begin();
      it != violations.end(); ++it){
    ret[*it] = m_type;
  }
  return ret;
}


/*
 * SATURATION REMAPPER
 */

SaturateRemapper::SaturateRemapper(const mat & bounding_box,
				   double fudge=PRETTY_SMALL) :
  m_bbox(bounding_box), m_fudge(fudge){
  assert(check_bounding_box(m_bbox));
  assert(m_fudge >= 0);
}


void SaturateRemapper::remapper(TypedPoints & points){
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(D == m_bbox.n_rows);
  assert(2 == m_bbox.n_cols);

  uvec normal_rows = points.get_normal_mask();
  NodeRemapRegistry reg;
  for(uint d = 0; d < D; ++d){
    uvec dim_col = uvec{d};
    double lb = m_bbox(d,0);
    double ub = m_bbox(d,1);
    
    uvec low_mask = find(points.col(d) < lb + m_fudge);
    points.m_points(normal_rows,dim_col).fill(lb + m_fudge);

    uvec high_mask = find(points.col(d) > ub - m_fudge);
    points.m_points(normal_rows,dim_col).fill(ub - m_fudge);
  }  
}


/*
 * WRAP REMAPPER
 */

WrapRemapper::WrapRemapper(const mat & bounding_box) :
  m_bbox(bounding_box){
  assert(check_bounding_box(m_bbox));
}


NodeRemapRegistry WrapRemapper::remapper(const TypedPoints & points){
  uint N = points.n_rows;
  uint D = points.n_cols;
  assert(D == m_bbox.n_rows);
  
  uvec normal_rows = points.get_normal_mask();
  NodeRemapRegistry reg;
  for(uint d = 0; d < D; ++d){
    if(!is_finite(bbox.row(d)))
      continue;
    
    uvec dim_col = uvec{d};
    double lb = m_bbox(d,0);
    double ub = m_bbox(d,1);

    vec tmp = points.col(d) - lb;
    tmp = vec_mod(tmp, ub - lb);
    points.col(d) = tmp + lb;
    
    assert(not any(points.col(d) > ub));
    assert(not any(points.col(d) < lb));
  }   
}


bool check_bounding_box(const mat & bbox){
  assert(2 == bbox.n_cols);
  assert(all(bbox.col(0) < bbox.col(1))); // Must be full dimensional
  return true;
}
