#include "discretizer.h"
#include "simulator.h"

#include <armadillo>

using namespace arma;
using namespace std;

mat model_q(const vec values,
	    const vector<sp_mat> p_blocks,
	    const mat costs,
	    const double gamma);

vec bellman_residual_from_model(const vec values,
				const vector<sp_mat> p_blocks,
				const mat costs,
				const double gamma);

vec bellman_residual_at_nodes(const Discretizer * disc,
			      const Simulator * sim,
			      const vec & values,
			      double gamma,
			      int steps = 0,
			      uint samples = 25);

vec bellman_residual_at_nodes(const TypedDiscretizer * disc,
			      const TypedSimulator * sim,
			      const vec & values,
			      double gamma,
			      int steps = 0,
			      uint samples = 25);

vec bellman_residual_at_centers(const Discretizer * disc,
				const Simulator * sim,
				const vec & values,
				double gamma,
				int steps = 0,
				uint samples = 25);

vec bellman_residual_at_centers_with_flows(const Discretizer * disc,
					   const Simulator * sim,
					   const vec & values,
					   const mat & flows,
					   double gamma,
					   int steps = 0,
					   uint samples = 25);

vec advantage_residual(const Discretizer * disc,
                       const Simulator * sim,
                       const vec & values,
                       double gamma,
                       uint samples = 25);

vec advantage_function(const Discretizer * disc,
                       const Simulator * sim,
                       const vec & values,
                       double gamma,
                       int steps = 0,
                       uint samples = 25);

uvec grad_policy(const Discretizer * disc,
                 const Simulator * sim,
                 const vec & value,
                 uint samples=25);

uvec q_policy(const Discretizer * disc,
             const Simulator * sim,
             const vec & values,
             double gamma,
             uint samples = 25);
uvec q_policy_at_nodes(const Discretizer * disc,
		       const Simulator * sim,
		       const vec & values,
		       double gamma,
		       uint samples = 25);

uvec flow_policy(const Discretizer * disc,
                 const mat & flow);

uvec flow_policy_diff(const Discretizer * disc,
                      const mat & flow);

uvec policy_agg(const Discretizer * disc,
                     const Simulator * sim,
                     const vec & values,
                     const mat & flows,
                     const double gamma,
                     const uint samples=25);

vec agg_flow_at_centers(const Discretizer * disc,
                        const mat & flows);

sp_mat build_markov_chain_from_blocks(const vector<sp_mat> & blocks,
				      const uvec policy);
