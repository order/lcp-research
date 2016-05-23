#include "mcts.h"
#include "rand.h"

MCTSNode::MCTSNode(const vec & state,
		   MCTSContext * context){

  // Store state and context information
  _id = _NODE_ID++;
  _state = state;
  _context = context;
  _n_actions = context->n_actions;

  // Init visits
  _total_visits = 0;
  _child_visits = zeros<uvec>(_n_actions);
  _ucb_scale = context->_ucb_scale;

  // Init Q estimate and costs cache
  _q = _q_fn.f(_state);
  _v = max(_q);
  _costs = context->_cost_fn.costs(_state,*context->actions);
  assert(_n_actions == _costs.n_elem);


  // Init action probabilities
  _p_scale = context->_p_scale;
  _prob = context->_prob_fn.p(state);
  assert(_n_actions == _prob.n_elem);

  _n_children = 0;
  _children.resize(_n_actions);
}

void MCTSNode::print_debug() const{
  // Basic debug information
  std::cout << "N[" << _id << "] = <"
	    << _state << ',' << _v << '>' << std::endl;
}

bool MCTSNode::is_leaf() const{
  return 0 == _n_children;
}

bool MCTSNode::has_unexplored() const{
  return _n_actions > _n_children;
}

double MCTSNode::get_action_ucb(uint a_idx) const{
  // Actually the "lower confidence bound" because we're minimizing cost

  return _q(a_idx)
    + _ucb_scale * sqrt(2.0 * log(_total_visits) / _child_visits(a_idx))
    + _p_scale * _prob(a_idx) / (1.0 + _child_visits(a_idx));
}


uint MCTSNode::get_best_action() const{
  assert(_n_actions > 0);
  uint a_idx = 0;
  double best_score = get_action_ucb(0);
  for(uint i = 1; i < _n_actions; i++){
    double score = get_action_ucb(i);
    if(score < best_score){
      best_score = score;
      a_idx = i;
    }
  }
  assert(isfinite(best_score));
  return a_idx;
}

MCTSNode* MCTSNode::pick_child(uint a_idx){
  /*
    Pick a child. If the number of children is less than
    log_2(visits + 1) + 1, then draw a new child.
  */
  uint n_nodes = _children[a_idx].size();
  uint n_visit = _child_visits[a_idx];
  // Use a log schedule
  bool new_node = (n_nodes < log2(n_visits + 1) + 1);
  assert(n_nodes == 0 ? n_nodes : true);

  if(new_node){
    return sample_new_node(uint a_idx);
  }
  else{
    std::uniform_int_distribution<uint> c_dist(0,n_nodes-1);
    uint c_idx = c_dist(MT_GEN);
    return _children[a_idx][c_idx];
  }  
}

MCTSNode* MCTSNode::get_best_child(){
  uint a_idx = get_best_action();
  return pick_child(a_idx);
}

MCTSNode * MCTSNode::sample_new_node(uint a_idx) const{
  vec action = _context->_actions.row(a_idx);
  vec state =  _context->_trans_fn.get_next_state(_state,action);

  return add_child(a_idx,state);
}

MCTSNode * add_child(uint a_idx, const vec & state){
  MCTSNode * find_res = find_node(a_idx,state);
  if(NULL != find_res){
    return find_res;
  }
  MCTSNode * new_child = new MCTSNode(state,_context);
  _context->master_list.push_back(new_child); // Add to master
  _children[a_idx].push_back(new_child); // Add to action list
  return new_child;
}

MCTSNode * find_child(uint a_idx, const vec & state, double thresh){
  for(ChildList::const_iterator it = _children[a_idx].begin();
      it != _children[a_idx].end(); ++it){
    double dist = dist(state,(*it)->_state);
    if(dist < thresh){
      return *it;
    }
  }
  return NULL;
}
