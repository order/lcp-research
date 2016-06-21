#ifndef __Z_STATE_FUNCTION_INCLUDED__
#define __Z_STATE_FUNCTION_INCLUDED__

#include <armadillo>
#include "discrete.h"
using namespace arma;

class RealFunction{
 public:
  virtual ~RealFunction();
  virtual vec f(const mat & points) const = 0;
  virtual double f(const vec & points) const = 0;
  virtual uint dom_dim() const = 0;
};

class MultiFunction{
 public:
  virtual ~MultiFunction();
  virtual mat f(const mat & points) const = 0;
  virtual vec f(const vec & points) const = 0;
  virtual uint dom_dim() const = 0;
  virtual uint range_dim() const = 0;
};

class InterpFunction : public RealFunction{
 public:
  InterpFunction(const vec & val,
		 const RegGrid & grid);
  ~InterpFunction();
  vec f(const mat & points) const;
  double f(const vec & points) const;
  uint dom_dim() const;
 
 protected:
  vec _val;
  RegGrid _grid;
};

class InterpMultiFunction : public MultiFunction{
 public:
  InterpMultiFunction(const mat & val,
		 const RegGrid & grid);
  ~InterpMultiFunction();
  mat f(const mat & points) const;
  vec f(const vec & points) const;
  uint dom_dim() const;
  uint range_dim() const;
 
 protected:
  mat _val;
  RegGrid _grid;
};

class ConstMultiFunction : public MultiFunction{
 public:
  ConstMultiFunction(uint n, double val);
  mat f(const mat & points) const;
  vec f(const vec & points) const;
  uint dom_dim() const;
  uint range_dim() const;
 
 protected:
  uint _n;
  double _val;
};

class ProbFunction : public InterpMultiFunction{
 public:
  ProbFunction(const mat & val,
	       const RegGrid & grid);
  mat f(const mat & points) const;
  vec f(const vec & points) const;
};

#endif
