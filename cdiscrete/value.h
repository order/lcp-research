#include <armadillo>

#include "discrete.h"
#include "policy.h"
#include "simulate.h"

using namespace arma;

class QPolicy : public DiscretePolicy{
 public:
  QPolicy(const mat & q,
	  const mat & actions,
	  const RegGrid & grid);
  mat get_actions(const mat & points) const;
  uvec get_action_indices(const mat & points) const;
  uint get_action_dim() const;
  
 protected:
  RegGrid _grid;
  mat _q;
  mat _actions;
};
