#include "simulate.h"

mat generate_q_values(const vec & values,
		      const RegGrid & grid,
		      const Problem & problem);

class QPolicy : public DiscretePolicy{
 public:
  QPolicy(const mat & actions,
	  const vec & values,
	  const RegGrid & grid,
	  const Problem & problem);
  mat get_next_states(const mat & points, const mat & actions) const;
  mat get_next_states(const mat & points, const mat & actions) const;

 protected:
  RegGrid _grid;
  mat _q_values;
  mat _actions;
}
