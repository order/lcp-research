#ifndef __MCTS_INCLUDED__
#define __MCTS_INCLUDED__

// List (indexed by action id) of lists of children
typedef vector<MCTSNode*> ChildList;

struct MCTSContext{
  uint _NODE_ID = 0;
  vector<MCTSNode*> _master_list;

  TransferFunction * trans_fn;
  CostFuction * cost_fn;
  double discount;

  QFunction * q_fn;
  ActionProbDist * prob_fn;
  Policy * rollout;
  
  mat * actions;
  uint n_actions;

  double p_scale; // weight for initial probability
  double ucb_scale; // weight for UCB term

};

class MCTSNode{
 public:
  MCTSNode(const vec & state,
	   MCTSContext * context);
  void print_debug() const;
  
  bool is_leaf() const;
  bool has_unexplored() const;
  
  double get_action_ucb(uint a_idx) const;
  uint get_best_action() const;
  
  MCTSNode * pick_child(uint a_idx);
  MCTSNode * get_best_child();
  MCTSNode * sample_new_node();
  MCTSNode * add_child(uint a_idx, vec & state);
  MCTSNode * find_child(uint a_idx, vec & state, double thresh=1e-6);
  
  double update_stats(uint a_id, double G);
  
 protected:
  // Identity
  uint _id; // Node ID
  vec _state;
  uint _n_actions;
  MCTSContext * _context;

  // Visit counts
  uint _total_visits;
  uvec _child_visits;
  double _ucb_scale;


  // Values and costs
  double _v;
  vec _q;
  vec _costs;

  // Initial probability
  double _p_scale;
  vec _prob;

  uint _n_children;
  vector<ChildList> _children;
};

#endif
