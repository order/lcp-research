#ifndef __POLICY_INCLUDED__
#define __POLICY_INCLUDED__

#include <aramdillo>

// Abstract policy class
class Policy{
 public:
  virtual vec get_actions(const mat & points);
}

// Bang-bang policy for 2D (x,v) double integrator
class  DIBangBangPolicy : public Policy{
 public:
  DIBangBangPolicy(const & mat actions);
  uvec get_action_indices(const mat & points);
  vec get_actions(const mat & points);
 protected:
  mat m_actions;
  uint m_n_actions;
}

class TransferFn{
 public:
  virtual vec get_next_states(const mat & points, const vec);
}

#endif
