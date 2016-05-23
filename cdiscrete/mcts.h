#ifndef __MCTS_INCLUDED__
#define __MCTS_INCLUDED_

// List (indexed by action id) of lists of children
typedef vector<vector<MCTSNode*>> ChildList;

class MCTSNode{
 public:
  MCTSNode(vec & state,
	   uint n_actions);
  void initialize();
  void print_debug() const;
  bool is_leaf() const;
  double get_action_ucb(uint a_idx) const;
  uint get_best_action() const;

  MCTSNode * get_best_child();
  MCTSNode * get_child(uint a_idx) const;
  MCTSNode * get_child(uint a_idx, uint c_idx) const;
  MCTSNode * find_node(vec & target, double dist_thresh) const;
  MCTSNode * find_node(vec & target, uint a_idx, double dist_thresh) const;

  void add_child(uint a_idx, MCTSNode * new_child);
  MCTSNode * sample_child(uint a_idx);

  double update_stats(uint a_id, double G);
  
 protected:
  // Identity
  uint _ID; // Node ID
  vec _state;
  uint _n_actions;
  mat * _actions;

  // Dynamics functions
  TransferFunction * _t_fn;
  CostFunction * _c_fn;
  ActionProbDist * _p_fn; // TODO
  double discount;
  // Visit counts
  uint _total_visits;
  uvec _child_visits;

  // Values and costs
  double _v;
  vec _q;
  vec _costs;

  // Initial probability
  double _p_scale;
  vec _prob;

  ChildList children;
};

#endif
