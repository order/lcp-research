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
  _context = context;
  _n_actions = context->n_actions;

  _fresh = true;

  // Init visits
  _total_visits = 0;
  _child_visits = zeros<uvec>(_n_actions);
  _ucb_scale = context->ucb_scale;

  // Init Q estimate and costs cache
  _q = context->q_fn->f(_state);
  _v = max(_q);
  mat costs = context->cost_fn->get_costs(_state.t(),
					  *context->actions);
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

void MCTSNode::print_debug() const{
  // Basic debug information

  vec u = get_all_ucbs();
  
  std::cout << "N" << _id << ":\n"
	    << "\tState:"<< _state.t()
	    << "\tQ:" << _q.t()
    	    << "\t\tv: " << _v << std::endl
	    << "\t\tcosts:" << _costs.t()
	    << "\tP:" << _prob.t()
	    << "\tVisits" << _child_visits.t()
	    << "\t\tTotal: " << _total_visits << std::endl
	    << "\tUCB:" << u.t()
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

uint MCTSNode::id() const{
  return _id;
}

bool MCTSNode::is_leaf() const{
  return 0 == _n_children;
}

bool MCTSNode::has_unexplored() const{
  return _n_actions > _n_children;
}

vec MCTSNode::get_all_ucbs() const{
  vec u = vec(_n_actions);
  for(uint a_idx = 0; a_idx < _n_actions; a_idx++){
    u(a_idx) = get_action_ucb(a_idx);
  }
  return u;
}

double MCTSNode::get_action_ucb(uint a_idx) const{
  // Actually the "lower confidence bound" because we're minimizing cost

  uint total_v = _total_visits+1; // +1 to avoid nan
  uint child_v = _child_visits(a_idx)+1;
  return _q(a_idx)
    - _ucb_scale * sqrt(2.0 * log(total_v) / child_v)
    - _p_scale * _prob(a_idx) / child_v;
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

  created = false;
  
  uint n_nodes = _children[a_idx].size();
  uint n_visits = _child_visits[a_idx];
  // Use a log schedule
  bool new_node = (n_nodes < log2(n_visits + 1) + 1);
  std::cout << log2(n_visits + 1) + 1
	    << " " << new_node << std::endl;
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

MCTSNode* MCTSNode::get_best_child(){
  uint a_idx = get_best_action();
  return pick_child(a_idx);
}

MCTSNode * MCTSNode::sample_new_node(uint a_idx){
  vec action = _context->actions->row(a_idx);
  vec state =  _context->trans_fn->get_next_state(_state,action);

  return add_child(a_idx,state);
}

// Add a new child (unless similar node exists)
MCTSNode * MCTSNode::add_child(uint a_idx, const vec & state){
  MCTSNode * find_res = find_child(a_idx,state);
  if(NULL != find_res){
    return find_res;
  }
  MCTSNode * new_child = new MCTSNode(state,_context);
  _context->master_list.push_back(new_child); // Add to master
  _children[a_idx].push_back(new_child); // Add to action list
  return new_child;
}

// See if state already exists as a child of the action,
// or if a similar node exists
MCTSNode * MCTSNode::find_child(uint a_idx, const vec & state, double thresh){
  for(NodeList::const_iterator it = _children[a_idx].begin();
      it != _children[a_idx].end(); ++it){
    double dst = norm(state - (*it)->_state);
    if(dst < thresh){
      return *it;
    }
  }
  return NULL;
}

// Whether just created or not.
// Used primarily in expand_tree to determine if dive is over
bool is_fresh() const{
  return _fresh;
}

// Sets the fresh flag to false;
void set_stale() const{
  _fresh = false;
}

// Adds new nodes to tree until out of budget
void grow_tree(MCTSNode * root, uint budget){
  for(uint b = 0; b < budget; b++){
    Path path = expand_tree(root);
    assert(path.size() > 0);
    double G = simulate_leaf(path);
    update_path(path,G);
  }
}

Path expand_tree(MCTSNode * root){
  Path path;  
  path.append(root);
  
  MCTSNode * curr = root;
  uint a_idx;
  for(uint i = 0; i < 2500; i++){
    a_idx = curr->get_best_action();
    curr = curr->get_best_child();
    path.second.append(a_idx);
    path.first.append(curr);

    // Check if the node was just created
    if(curr.is_fresh()){
      curr.set_stale();
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
  
}
void update_path(const MCTSPath & path, double gain){
  NodeList nodes = path.first;
  ActionList actions = path.second;
  assert(nodes.size() = actions.size();
  
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
    std::cout << "Deleting: N" << (*it)->id() << std::endl;
    delete *it;
    *it = NULL;
  }
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
  uint A = root->_context->n_actions;

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
    if(depth > max_depth){continue;}

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

