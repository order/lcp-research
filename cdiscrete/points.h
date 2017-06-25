#ifndef __Z_POINTS_INCLUDED__
#define __Z_POINTS_INCLUDED__

#include <armadillo>
#include <map>

#define SPATIAL_TYPE 0
#define SPECIAL_FILL arma::datum::nan

typedef std::map<uint,uint> TypeRegistry;
typedef arma::mat Points; // Basic untyped points

/*****************************************************
 * TYPING RULES *
 ****************/
class TypeRule{
 public:
  /*
    Apply rule to point set. Any special points found are returned in a
    map from indices to types.
  */
  virtual TypeRegistry type_elements(const Points & points) const = 0;
};
typedef std::vector<TypeRule> TypeRuleList;

class OutOfBoundsRule : public TypeRule{
 public:
  OutOfBoundsRule(const arma::mat & bounding_box, uint oob_type);
  TypeRegistry type_elements(const Points & points) const;
 protected:
  arma::mat m_bbox;
  uint m_type;
};


class NodeRemapper{
 public:
  /*
    Apply remapping rule to point set. 
    Any remapped points found are returned in a map from indices to new 
    vectors.
  */  
  virtual void remap(Points & points) const = 0;
};
typedef std::vector<NodeRemapper> NodeRemapperList;


class SaturateRemapper : public NodeRemapper{
  SaturateRemapper(const arma::mat & bounding_box);
  void remapper(Points & points);
  
 protected:
  arma::mat m_bbox;
};

class WrapRemapper : public NodeRemapper{
  WrapRemapper(const arma::mat & bounding_box);
  void remapper(Points & points);
  
 protected:
  arma::mat m_bbox;
};

/************************************************************
 * POINT TYPE OBJECT *
 *********************/

class TypedPoints{
  /*
    Class adding typing information to organize points. Points are either 
    bounded Euclidean points, or are "special" (e.g. out of bound and other 
    sink states).
    Euclidean points have type EUCLIDEAN_TYPE. Other point types are specially
    registered.
    Rules for detecting and applying rules can be applied as vectors.

    NB: The assumption is that most nodes are Euclidean and "normal".
  */
 public:
  TypedPoints(const Points & points, const TypeRegistry & reg);
  TypedPoints(const Points & points);
  TypedPoints();

  // Registry functions
  uint get_next_type(); // Max registry keys + 1
  void register_type(uint idx, uint ntype); // Add new element to registry
  uint num_special_nodes() const;
  uint num_spatial_nodes() const;
  uint num_all_nodes() const;

  arma::uvec get_normal_mask() const;
  arma::uvec get_special_mask() const;

  // Run rules for typing and remapping.
  void apply_typing_rule(const TypeRule & rule);
  void apply_typing_rules(const TypeRuleList & rules);
  void apply_remapper(const NodeRemapper & remapper);
  void apply_remappers(const NodeRemapperList & remappers);
  
  Points m_points;
  TypeRegistry m_reg;

  bool check_validity() const;
  bool check_bounding_box(arma::vec & low, arma::vec & high) const;

 protected:
  void _ensure_blanked();
};


// MISC FUNCTIONS

bool check_bounding_box(const arma::mat & bbox);


#endif
