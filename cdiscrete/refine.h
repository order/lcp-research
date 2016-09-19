#include "discretizer.h"
#include "simulator.h"

#include <armadillo>

using namespace arma;
using namespace std;

vec bellman_residual(const Discretizer * disc,
                     const Simulator * sim,
                     const vec & values,
                     double gamma,
                     int steps = 0,
                     uint samples = 25);
vec bellman_residual_with_flows(const Discretizer * disc,
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

uvec flow_policy(const Discretizer * disc,
                 const mat & flow);

uvec flow_policy_diff(const Discretizer * disc,
                      const mat & flow);
