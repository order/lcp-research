#include "points.h"

using arma;
using std;

Points::Points(const mat & points, const NodeTypeRegistry & reg) :
  m_points(points), m_reg(reg), m_max_type(EUCLIDEAN_TYPE) {
  assert(m_points.n_rows >= m_reg.size());
  _ensure_blanked();
}

Points::Points(const mat & points) :
  m_points(points), m_max_type(EUCLIDEAN_TYPE) {}

Points::Points() : m_max_type(EUCLIDEAN_TYPE){}

uint Points::get_next_type(){
  return ++m_max_type;
}

void Points::register(Index idx, NodeType node_type){
  assert(node_type <= m_max_type);
  assert(idx <= m_points.n_rows);
  
  m_reg[idx] = node_type;
  m_points.row(idx).fill(SPECIAL_FILL);  // Blank out row
}

uint Points::num_special_nodes(){
  return m_reg.size();
}

uint Points::num_normal_nodes(){
  assert(m_points.n_rows >= m_reg.size());  
  return m_points.n_rows - m_reg.size();
}

uint Points::num_all_nodes(){
  return m_points.n_rows();
}

void Points::apply_typing_rule(const NodeTypeRule & rule){
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

void Points::apply_typing_rules(const NodeTypeRuleList & rules){
  /*
    Iterate through list and apply node type rules
  */
  auto bound_fn = bind(&Points::apply_typing_rule,
		       this,
		       std::placeholders::_1);
  for_each(rules.begin(), rules.end(), bound_fn);
}

void Points::apply_remapper(const NodeRemapper & remapper){
  NodeRemapRegistry new_points = remapper->remap(m_points);
  for(NodeRemapRegistry point_it = new_points.begin();
      point_it != new_points.end(); ++point_it){
    Index row_idx = point_it->first;
    vec new_point = point_it->second;
    assert(!m_points.row(row_idx).has_nan());  // Check not NaN

    m_points.row(row_idx) = new_point;  // Update row
  }
}

void Points::apply_remappers(const NodeRemapperList & remappers){
  /*
    Iterate through list and apply node remappers
  */
  auto bound_fn = bind(&Points::apply_remapper,
		       this,
		       std::placeholders::_1);
  for_each(rules.begin(), rules.end(), bound_fn);
}
