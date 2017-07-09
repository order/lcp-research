#ifndef __Z_POINTS_INCLUDED__
#define __Z_POINTS_INCLUDED__

#include <armadillo>
#include <map>
#include <memory>

typedef std::map<uint,uint> TypeRegistry;
typedef arma::mat Points; // Basic untyped points

class TypedPoints; // Forward declaration.

arma::uvec get_spatial_rows(const Points & points);
arma::uvec get_special_rows(const Points & points);
bool check_points(const Points & points);
bool check_bbox(const arma::mat & bbox);
bool check_points_in_bbox(const Points & points, const arma::mat & bbox);
bool check_points_in_bbox(const TypedPoints & points, const arma::mat & bbox);


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
typedef std::vector<std::unique_ptr<TypeRule>> TypeRuleList;

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
  virtual void remap(TypedPoints & points) const = 0;

};
typedef std::vector<std::unique_ptr<NodeRemapper>> NodeRemapperList;


class SaturateRemapper : public NodeRemapper{
 public:
  SaturateRemapper(const arma::mat & bounding_box);
  void remap(Points & points) const;
  void remap(TypedPoints & points) const;
  
 protected:
  arma::mat m_bbox;
};

class WrapRemapper : public NodeRemapper{
 public:
  WrapRemapper(const arma::mat & bounding_box);
  void remap(Points & points) const;
  void remap(TypedPoints & points) const;
 
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
  static const uint DEFAULT_OOB_TYPE = 1;
  static const uint SPATIAL_TYPE = 0;
  const double SPECIAL_FILL = arma::datum::nan; // Can't be static, apparently

  TypedPoints(const TypedPoints & other);  // Copy constructor
  TypedPoints(const Points & points, const TypeRegistry & reg);
  TypedPoints(const Points & points);
  TypedPoints();

  // Registry functions
  uint get_next_type(); // Max registry keys + 1
  void register_type(uint idx, uint ntype); // Add new element to registry
  uint num_special_nodes() const;
  uint num_spatial_nodes() const;
  uint num_all_nodes() const;

  arma::uvec get_spatial_mask() const;
  arma::uvec get_special_mask() const;
  bool is_special(uint idx) const;

  // Run rules for typing and remapping.
  void apply_typing_rule(const TypeRule & rule);
  void apply_typing_rules(const TypeRuleList & rules);
  void apply_remapper(const NodeRemapper & remapper);
  void apply_remappers(const NodeRemapperList & remappers);
  
  Points m_points;
  TypeRegistry m_reg;
  uint n_rows;
  uint n_cols;

  bool check_validity() const;
  bool check_in_bbox(const arma::mat & bbox) const;
  bool check_in_bbox(const arma::vec & low, const arma::vec & high) const;
  bool equals(const TypedPoints & other) const;
  friend std::ostream& operator<<(std::ostream& os, const TypedPoints& p);  

 protected:
  void _ensure_blanked();
};


// MISC FUNCTIONS




#endif
