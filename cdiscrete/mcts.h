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
#include "simulate.h"
#include "transfer.h"

#define Q_AVG     1
#define Q_EXP_AVG 2

#define UPDATE_RET_V    1
#define UPDATE_RET_Q    2
#define UPDATE_RET_GAIN 4

using namespace arma;

class MCTSNode; //forward declare

// List (indexed by action id) of lists of children
typedef std::vector<MCTSNode*> NodeList;
typedef std::vector<uint> ActionList;
typedef std::pair<NodeList,ActionList> Path;

struct MCTSContext{
  uint _NODE_ID = 0;
  std::vector<MCTSNode*> master_list;

  Problem * problem_ptr;
  uint n_actions;

  RealFunction * v_fn;
  MultiFunction * q_fn;
  ProbFunction * prob_fn;
  DiscretePolicy * rollout;

  double p_scale; // weight for initial probability
  double ucb_scale; // weight for UCB term
  uint rollout_horizon;

  double init_q_mult; // Multiplier for init Q estimates
  uint q_update_mode;
  double q_stepsize; // Multiplier if Q_EXP_AVG is true
  uint update_ret_mode;
};

class MCTSNode{
 public:
  MCTSNode(const vec & state,
	   MCTSContext * context);
  ~MCTSNode();
  void print_debug() const;

  uint get_id() const;
  vec get_state() const;
  
  bool is_leaf() const;
  bool has_unexplored() const;

  double get_nudge(uint a_idx) const;
  vec get_all_nudges() const;
  vec get_all_ucbs() const;
  double get_action_ucb(uint a_idx) const;
  uint get_best_action() const;
  uint get_freq_action() const;
  
  MCTSNode * pick_child(uint a_idx);
  MCTSNode * get_best_child();
  MCTSNode * sample_new_node(uint a_idx);
  MCTSNode * add_child(uint a_idx, const vec & state);
  MCTSNode * find_child(uint a_idx, const vec & state, double thresh=1e-15);
  double update(uint a_idx, double gain);

  bool is_fresh() const;
  void set_stale();
  
  double update_status(uint a_id, double G);

  friend void write_dot_file(std::string filename, MCTSNode * root);
  friend void grow_tree(MCTSNode * root, uint budget);
  friend Path find_path_and_make_leaf(MCTSNode * root);
  friend double simulate_leaf(Path & path);
  friend void update_path(const Path & path, double gain);

 protected:
  // Identity
  uint _id; // Node ID
  vec _state; 
  MCTSContext * _context_ptr;
  uint _n_actions;
  double _discount;

  bool _fresh;

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

  double _q_stepsize;

  uint _n_children;
  std::vector<NodeList> _children;
};
void add_root(MCTSContext * context, MCTSNode * root);
void delete_tree(MCTSContext * context);
void delete_context(MCTSContext * context);


void print_nodes(const MCTSContext & context);
string node_name(uint id);
string action_name(uint id, uint a_idx);
#endif
