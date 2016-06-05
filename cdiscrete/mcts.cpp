#include <assert.h>
#include <sstream>
#include <string>

#include "mcts.h"
#include "misc.h"

#include "boost/graph/graphviz.hpp"
#include "boost/graph/adjacency_list.hpp"


MCTSNode::MCTSNode(const vec & state,
		   MCTSContext * context){

  // Store state and context information
  _id = context->_NODE_ID++;
  _state = state;
  _context_ptr = context;
  _n_actions = context->n_actions;
  _discount = context->problem_ptr->discount;

  _fresh = true;

  // Init visits
  _total_visits = 0;
  _child_visits = zeros<uvec>(_n_actions);
  _ucb_scale = context->ucb_scale;

  // Init Q estimate and costs cache
  //_q = context->init_q_mult*context->q_fn->f(_state);
  _q = zeros<vec>(_n_actions);
  _v = min(_q);
  mat costs = context->problem_ptr
    ->cost_fn->get_costs(_state.t(),
			 context->problem_ptr->actions);
  assert(1 == costs.n_rows);
  assert(_n_actions == costs.n_cols);
  _costs = costs.row(0).t();


  // Init action probabilities
  _p_scale = context->p_scale;
  _prob = context->prob_fn->f(state);
  assert(_n_actions == _prob.n_elem);

  _n_children = 0;
  _children.resize(_n_actions);
}

MCTSNode::~MCTSNode(){
  for(vector<NodeList>::iterator it = _children.begin();
      it != _children.end(); it++){
    it->clear();
  }
  _children.clear();
}

void MCTSNode::print_debug() const{
  // Basic debug information

  vec u = get_all_ucb_scores();
  vec nudge = get_all_nudges();
  
  std::cout << "N" << _id << ":\n"
	    << "\tState:"<< _state.t()
	    << "\tQ:" << _q.t()
    	    << "\t\tv: " << _v << std::endl
	    << "\t\tcosts:" << _costs.t()
	    << "\tP:" << _prob.t()
	    << "\tVisits" << _child_visits.t()
	    << "\t\tTotal: " << _total_visits << std::endl
	    << "\tUCB:" << u.t()
    	    << "\t\tNudges: " << nudge.t() << std::endl
	    << "\tChildren:" << std::endl;
  for(uint a_idx = 0; a_idx < _n_actions; a_idx++){
    std::cout << "\t\ta" << a_idx << ": [";
    for(NodeList::const_iterator it = _children[a_idx].begin();
	it != _children[a_idx].end(); it++){
      if(it != _children[a_idx].begin()){
	std::cout << ',';
      }
      std::cout << 'N' << (*it)->_id;
    }
    std::cout << ']' << std::endl;
  }
}

uint MCTSNode::get_id() const{
  return _id;
}

vec MCTSNode::get_state() const{
  return _state;
}

bool MCTSNode::is_leaf() const{
  return 0 == _n_children;
}

bool MCTSNode::has_unexplored() const{
  return _n_actions > _n_children;
}

vec MCTSNode::get_all_ucb_scores() const{
  vec u = vec(_n_actions);
  for(uint a_idx = 0; a_idx < _n_actions; a_idx++){
    u(a_idx) = get_ucb_score(a_idx);
  }
  return u;
}

vec MCTSNode::get_all_nudges() const{
  vec u = vec(_n_actions);
  for(uint a_idx = 0; a_idx < _n_actions; a_idx++){
    u(a_idx) = get_nudge(a_idx);
  }
  return u;
}

double MCTSNode::get_nudge(uint a_idx) const{
    uint total_v = _total_visits+1; // +1 to avoid nan
    uint child_v = _child_visits(a_idx)+1;
    return  -_ucb_scale * sqrt(2.0 * log(total_v) / child_v)
      - _p_scale * _prob(a_idx) / child_v;
}

double MCTSNode::get_ucb_score(uint a_idx) const{
  // Actually the "lower confidence bound" because we're minimizing cost

  uint total_v = _total_visits+1; // +1 to avoid nan
  uint child_v = _child_visits(a_idx)+1;
  return _q(a_idx)
    - _ucb_scale * sqrt(2.0 * log(total_v) / child_v)
    - _p_scale * _prob(a_idx) / child_v;
}

uint MCTSNode::get_action(uint mode) const{
  if(mode == ACTION_Q){
    return get_q_action();
  }
  if(mode == ACTION_FREQ){
    return get_freq_action();
  }
  if(mode == ACTION_ROLLOUT){
    return _context_ptr->rollout->get_action_index(_state);
  }
  if(mode == ACTION_UCB){
    return get_ucb_action();
  }
  assert(false);
}

uint MCTSNode::get_q_action() const{
  uint a_idx = argmin(_q);
  assert(isfinite(_q(a_idx)));
  return a_idx;
}

uint MCTSNode::get_ucb_action() const{
  vec u = get_all_ucb_scores();
  uint a_idx = argmin(u);
  assert(isfinite(u(a_idx)));
  return a_idx;
}

uint MCTSNode::get_freq_action() const{
  uint a_idx = argmax(_child_visits);
  return a_idx;
}

MCTSNode* MCTSNode::pick_child(uint a_idx){
  /*
    Pick a child. If the number of children is less than
    log_2(visits + 1) + 1, then draw a new child.
  */
  
  uint n_nodes = _children[a_idx].size();
  uint n_visits = _child_visits[a_idx];
  // Use a log schedule
  bool new_node = (n_nodes < log(n_visits + 1) / log(10) + 1);
  
  assert((n_nodes == 0) ? new_node : true);

  if(new_node){
    return sample_new_node(a_idx);
  }
  else{
    std::uniform_int_distribution<uint> c_dist(0,n_nodes-1);
    uint c_idx = c_dist(MT_GEN);
    return _children[a_idx][c_idx];
  }  
}

MCTSNode * MCTSNode::sample_new_node(uint a_idx){
  vec action = _context_ptr->problem_ptr->actions.row(a_idx);
  vec state =  _context_ptr->problem_ptr
    ->trans_fn->get_next_state(_state,action);
  return add_child(a_idx,state);
}

// Add a new child (unless similar node exists)
MCTSNode * MCTSNode::add_child(uint a_idx, const vec & state){
  MCTSNode * find_res = find_child(a_idx,state);
  if(NULL != find_res){
    return find_res;
  }
  MCTSNode * new_child = new MCTSNode(state,_context_ptr);
  _context_ptr->master_list.push_back(new_child); // Add to master
  _children[a_idx].push_back(new_child); // Add to action list
  return new_child;
}

// See if state already exists as a child of the action,
// or if a similar node exists
MCTSNode * MCTSNode::find_child(uint a_idx,
				const vec & state,
				double thresh){
  for(NodeList::const_iterator it = _children[a_idx].begin();
      it != _children[a_idx].end(); ++it){
    double dst = norm(state - (*it)->_state);
    if(dst < thresh){
      return *it;
    }
  }
  return NULL;
}

double MCTSNode::update(uint a_idx,double gain){
  assert(!_fresh);
  _total_visits++;
  _child_visits(a_idx)++;

  // Child got G, so we got c[a] + d * G
  gain = _costs(a_idx) + _discount * gain;

  double alpha = max(1.0 / _child_visits(a_idx),
		     _context_ptr->q_min_step);
  _q(a_idx) *= (1.0 - alpha);
  _q(a_idx) += alpha * gain;  
  _v = min(_q);

  uint update_ret_mode = _context_ptr->update_ret_mode;
  if(update_ret_mode == UPDATE_RET_V){
    return _v;
  }
  if(update_ret_mode == UPDATE_RET_Q){
    return _q(a_idx);
  }
  if (update_ret_mode == UPDATE_RET_GAIN){
    return gain;
  }
  assert(false);
  return gain; // Return updated gain
}

// Whether just created or not.
// Used primarily in expand_tree to determine if dive is over
bool MCTSNode::is_fresh() const{
  return _fresh;
}

// Sets the fresh flag to false;
void MCTSNode::set_stale(){
  _fresh = false;
}

//===========================================================
// GROW TREE
// Adds new nodes to tree until out of budget
void grow_tree(MCTSNode * root, uint budget){
  for(uint b = 0; b < budget; b++){
    Path path = find_path_and_make_leaf(root); //Find leaf and add node
    double G = simulate_leaf(path); //Simulate out from the leaf
    update_path(path,G); //Update stats on nodes in the path
  }
}

// Follow path of best children until a new node is created
Path find_path_and_make_leaf(MCTSNode * root){
  Path path;  
  path.first.push_back(root);
  
  MCTSNode * curr = root;
  uint a_idx;
  for(uint i = 0; i < 2500; i++){
    a_idx = curr->get_ucb_action(); // From UCB+ score
    curr = curr->pick_child(a_idx); // Node may or may not be created

    // Add to path
    path.first.push_back(curr);
    path.second.push_back(a_idx);
    
    // Check if the node was just created
    if(curr->is_fresh()){
      // Fresh == just created
      curr->set_stale();
      return path;
    }
  }
  
  // Should never get here
  assert(false);
}

double simulate_leaf(Path & path){
  //Use rollout policy to simulate from state.
  // Count as a "visit"
  // Add first action to path's action list
  NodeList nodes = path.first;
  ActionList actions = path.second;
  // Node->action->N-> ... -> a->N
  assert(nodes.size() == actions.size()+1);
  
  MCTSNode * leaf = nodes.back();
  MCTSContext * context = leaf->_context_ptr;
  
  SimulationOutcome outcome;
  vec point = leaf->get_state();
  
  uint a_idx = context->rollout->get_action_index(point);
  path.second.push_back(a_idx);
  vec final_point;
  double gain = simulate_single(point,
				*context->problem_ptr,
				*context->rollout,
				context->rollout_horizon,
				final_point);
  double v = context->v_fn->f(final_point);

  // Gain = [1,d,d^2,d^3,...,d^{H-1}].T * [c_0,c_1,...,c_{H-1}] 
  double discount = context->problem_ptr->discount;
  // Tail Estimate: d^H * v(final_state)
  double v_discount = pow(discount,context->rollout_horizon);

  /*
  std::cout << "Found:\n"
	    << "\tGain:\t" << gain
	    << "\n\tTail state:\t" << final_point
	    << "\n\tTail:\t" << v_discount << " * " << v
	    << "\n\tTotal:\t" << gain + v_discount * v << std::endl;
  */
  
  return gain + v_discount * v;
}

void update_path(const Path & path, double gain){
  NodeList nodes = path.first;
  ActionList actions = path.second;
  // Node->action->N-> ... -> a->N->a
  assert(nodes.size() > 0);
  assert(nodes.size() == actions.size());
  uint L = nodes.size();
  
  for(int l = L-1; l >= 0; l--){
    gain = nodes[l]->update(actions[l], gain);
  }  
}

// Add root to context
void add_root(MCTSContext * context, MCTSNode * root){
  root->set_stale(); // Root is "explored" already.
  context->master_list.push_back(root);
}

// Delete all nodes (all nodes should be created by "new"
// e.g. from add_child
void delete_tree(MCTSContext * context){
  for(vector<MCTSNode*>::iterator it = context->master_list.begin();
      it != context->master_list.end(); it++){
    delete *it;
    *it = NULL;
  }
  context->master_list.clear();
}

// Delete all nodes (all nodes should be created by "new"
// e.g. from add_child
void delete_context(MCTSContext * context){
  delete_tree(context);
  delete context->v_fn;
  delete context->q_fn;
  delete context->prob_fn;
  delete context->rollout;

}

//===================================
// FUNCTIONS FOR DOT PRINTING

string node_name(uint id){
  stringstream ss;
  ss << "N" << id;
  return ss.str();
}

string action_name(uint id,uint a_idx){
  stringstream ss;
  ss << "a_" << id << "_" << a_idx;
  return ss.str();
}

void print_nodes(const MCTSContext & context){
  for(std::vector<MCTSNode*>::const_iterator it = context.master_list.begin();
      it != context.master_list.end(); it++){
    (*it)->print_debug();
  }
}

void write_dot_file(std::string filename, MCTSNode * root){
  uint max_depth = 3;
  uint A = root->_context_ptr->n_actions;

  typedef std::pair<MCTSNode*,uint> f_elem;
  vector<f_elem> fringe;
  fringe.push_back(make_pair(root,0));
  
  ofstream fh;
  fh.open(filename);
  fh << "digraph mcts_tree {" << std::endl;

  while(fringe.size() > 0){
    f_elem curr_elem = fringe.back();
    fringe.pop_back();

    MCTSNode * curr_node = curr_elem.first;
    uint depth = curr_elem.second;
    if(depth >= max_depth){
      fh << "\t" << node_name(curr_node->_id)
	 << " [shape=box,label=\"...\"];" << std::endl;
      continue;
    }

    // Build node
    fh << "\t" << node_name(curr_node->_id) << " [shape=box];" << std::endl;
    for(uint a_idx = 0; a_idx < A; a_idx++){
      for(NodeList::const_iterator it
	    = curr_node->_children[a_idx].begin();
	  it != curr_node->_children[a_idx].end(); it++){
	if(it == curr_node->_children[a_idx].begin()){
	  // Build action node:
	  fh << "\t" << action_name(curr_node->_id,a_idx)
	     << " [shape=diamond,label=\"A"
	     << a_idx << "\"];" << std::endl;
	  
	  // Connect parent to action
	  fh << "\t" << node_name(curr_node->_id)
	     << " -> "
	     << action_name(curr_node->_id,a_idx) <<";"<< std::endl;
	}
	// Connect action to child
	fh << "\t" << action_name(curr_node->_id,a_idx)
	   << " -> "
	   << node_name((*it)->_id) << ";" << std::endl;
	fringe.push_back(make_pair(*it,depth+1));
      }
    }
  }
  
  fh << "}" << std::endl;
  fh.close();
}

