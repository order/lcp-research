#ifndef __Z_MARSHAL_INCLUDED__
#define __Z_MARSHAL_INCLUDED__

#include <armadillo>
#include <string>

#define H_NUM_OBJS 0
#define H_HEADER_SZ 1
#define H_DATA_SZ 2

using namespace arma;

// Bundles stuff up into a single vector and
// writes to binary file
class Marshaller{
 public:
  Marshaller();
  void add_scalar(double s);
  void add_vec(const vec & v);
  void add_mat(const mat & A);
  
  void save(const std::string & filename) const;
  void clear();
 
  uvec _header;
  uint h_idx; // Header index
  
  vec _data;
  uint d_idx; // Data index
};

// Reads vector in as binary file.
class Demarshaller{
 public:
  Demarshaller(const std::string & filename);
  void load(const std::string & filename);
  void reset();
  void skip();
  void set_pos(uint idx);
  void clear();

  uint get_num_objs() const;

  double get_scalar();
  vec get_vec();
  mat get_mat();

  //Verbose versions
  double get_scalar(const std::string & field);
  vec get_vec(const std::string & field);
  mat get_mat(const std::string & field);
  
  uvec _header;
  uint h_idx; // Header index
  
  vec _data;
  uint d_idx; // Data index
};

#endif
