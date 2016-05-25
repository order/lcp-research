#ifndef __MCTS_INCLUDED__
#define __MCTS_INCLUDED__

#include <armadillo>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "costs.h"
#include "function.h"
#include "policy.h"
#include "transfer.h"


using namespace arma;

class MCTSNode; //forward declare

// List (indexed by action id) of lists of children
typedef std::vector<MCTSNode*> ChildList;

struct MCTSContext{
  uint _NODE_ID = 0;
  std::vector<MCTSNode*> master_list;

  TransferFunction * trans_fn;
  CostFunction * cost_fn;
  double discount;

  MultiFunction * q_fn;
  ProbFunction * prob_fn;
  Policy * rollout;
  
  mat * actions;
  uint n_actions;

  double p_scale; // weight for initial probability
  double ucb_scale; // weight for UCB term

};

void print_nodes(const MCTSContext & context);

string node_name(uint id);
string action_name(uint id, uint a_idx);
//void write_dot_file(std::string filename, MCTSNode * root);


class MCTSNode{
 public:
  MCTSNode(const vec & state,
	   MCTSContext * context);
  void print_debug() const;
  
  bool is_leaf() const;
  bool has_unexplored() const;

  vec get_all_ucbs() const;
  double get_action_ucb(uint a_idx) const;
  uint get_best_action() const;
  
  MCTSNode * pick_child(uint a_idx);
  MCTSNode * get_best_child();
  MCTSNode * sample_new_node(uint a_idx);
  MCTSNode * add_child(uint a_idx, const vec & state);
  MCTSNode * find_child(uint a_idx, const vec & state, double thresh=1e-6);
  
  double update_status(uint a_id, double G);

  friend void write_dot_file(std::string filename, MCTSNode * root);


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
  std::vector<ChildList> _children;
};

#endif
