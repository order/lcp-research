#include "marshaller.h"
#include <assert.h>

Marshaller::Marshaller(){
  clear();
}

void Marshaller::clear(){
    // Set up structures for writing
    _header = zeros<uvec>(3);
    _header(H_HEADER_SZ) = 3;
    _data.clear();
    h_idx = 3;
    d_idx = 0;
}

void Marshaller::add_scalar(double d){  
  _header(H_NUM_OBJS)  += 1;
  _header(H_HEADER_SZ) += 1;
  _header(H_DATA_SZ)   += 1;

  _header.resize(_header(H_HEADER_SZ));
  _data.resize(_header(H_DATA_SZ));

  _header(h_idx++) = 0;
  _data(d_idx++) = d;
  
  assert(_header(H_HEADER_SZ) == h_idx);
  assert(_header(H_DATA_SZ) == d_idx);
}

void Marshaller::add_vec(const vec & v){
  _header(H_NUM_OBJS)  += 1;
  _header(H_HEADER_SZ) += 2;
  _header(H_DATA_SZ)   += v.n_elem;
  
  _header.resize(_header(H_HEADER_SZ));
  _data.resize(_header(H_DATA_SZ));

  _header(h_idx++) = 1;
  _header(h_idx++) = v.n_elem;
  _data.tail(v.n_elem) = v;
  d_idx += v.n_elem;

  assert(_header(H_HEADER_SZ) == h_idx);
  assert(_header(H_DATA_SZ) == d_idx);
}

void Marshaller::add_mat(const mat & A){
  _header(H_NUM_OBJS)  += 1;
  _header(H_HEADER_SZ) += 3;
  _header(H_DATA_SZ)   += A.n_elem;
  
  _header.resize(_header(H_HEADER_SZ));
  _data.resize(_header(H_DATA_SZ));

  _header(h_idx++) = 2;
  _header(h_idx++) = A.n_rows;
  _header(h_idx++) = A.n_cols;
  _data.tail(A.n_elem) = vectorise(A); // Fortran order
  d_idx += A.n_elem;

  assert(_header(H_HEADER_SZ) == h_idx);
  assert(_header(H_DATA_SZ) == d_idx);
}

void Marshaller::add_cube(const cube & C){
  _header(H_NUM_OBJS)  += 1;
  _header(H_HEADER_SZ) += 4;
  _header(H_DATA_SZ)   += C.n_elem;
  
  _header.resize(_header(H_HEADER_SZ));
  _data.resize(_header(H_DATA_SZ));

  _header(h_idx++) = 3;
  _header(h_idx++) = C.n_rows;
  _header(h_idx++) = C.n_cols;
  _header(h_idx++) = C.n_slices;
  _data.tail(C.n_elem) = vectorise(C); // Fortran order
  d_idx += C.n_elem;

  assert(_header(H_HEADER_SZ) == h_idx);
  assert(_header(H_DATA_SZ) == d_idx);
}


void Marshaller::save(const std::string & filename) const{
  assert(_header.n_elem == _header(H_HEADER_SZ));
  assert(_data.n_elem == _header(H_DATA_SZ));
  
  vec output = vec(_header.n_elem + _data.n_elem);
  output.head(_header.n_elem) = conv_to<vec>::from(_header);
  output.tail(_data.n_elem) = _data;

  output.save(filename,raw_binary);
}

Demarshaller::Demarshaller(const std::string & filename){
  load(filename);
}

void Demarshaller::load(const std::string & filename){
  vec input;
  input.load(filename);
  uint H = input(H_HEADER_SZ);
  uint D = input(H_DATA_SZ);
  
  assert(input.n_elem == H + D);
  _header = conv_to<uvec>::from(input.head(H));
  _data = input.tail(D);
  h_idx = 3;
  d_idx = 0;
}

void Demarshaller::reset(){
  h_idx = 3;
  d_idx = 0;
}

void Demarshaller::skip(){
  uint obj_dim = _header[h_idx++];
  if(0 == obj_dim){
    d_idx++;
    return;
  }

  uint data_len = 1;
  for(uint i = 0; i < obj_dim;i++){
    data_len *= _header[h_idx++];
  }
  d_idx += data_len; 
}

void Demarshaller::set_pos(uint idx){
  reset();
  for(uint i = 0; i < idx; i++){
    skip();
  }
}

uint Demarshaller::get_num_objs() const{
  return _header(H_NUM_OBJS);
}

void Demarshaller::clear(){
  _header.clear();
  h_idx = 0;
  _data.clear();
  d_idx = 0;
}

double Demarshaller::get_scalar(){
  assert(0 == _header[h_idx]);
  h_idx++;
  
  return _data(d_idx++);
}

vec Demarshaller::get_vec(){
  assert(1 == _header[h_idx]); // Should start on dim
  uint N = _header[++h_idx]; // Get len
  h_idx++; // Increment to next start

  vec v = _data(span(d_idx,d_idx+N-1)); // Read in span
  assert(N == v.n_elem);
  d_idx += N;

  return v;
}

mat Demarshaller::get_mat(){
  assert(2 == _header[h_idx]); // Should start on dim
  uint R = _header[++h_idx]; // Row
  uint C = _header[++h_idx]; // Col
  h_idx++; // Increment to next start

  mat A = conv_to<mat>::from(_data(span(d_idx,d_idx+R*C-1)));
  assert(R*C == A.n_elem);
  d_idx += R*C;
  
  return reshape(A,R,C);
}

cube Demarshaller::get_cube(){
  assert(3 == _header[h_idx]); // Should start on dim
  uint R = _header[++h_idx]; // Row
  uint C = _header[++h_idx]; // Col
  uint S = _header[++h_idx]; // Slice
  h_idx++; // Increment to next start

  cube c = cube(R*C*S,1,1);
  c.slice(0).col(0) = _data(span(d_idx,d_idx+R*C*S-1));
  d_idx += R*C*S;
  
  return reshape(c,R,C,S);
}


double Demarshaller::get_scalar(const std::string & field,bool verbose){
  double x = get_scalar();
  if(verbose)
    std::cout << field << ": " << x << std::endl;
  return x;
}
vec Demarshaller::get_vec(const std::string & field,bool verbose){
  vec x = get_vec();
  if(verbose)
    std::cout << field << ": " << size(x) << std::endl;
  return x;
}
mat Demarshaller::get_mat(const std::string & field,bool verbose){
  mat x = get_mat();
  if(verbose)
    std::cout << field << ": " << size(x) << std::endl;
  return x;
}

cube Demarshaller::get_cube(const std::string & field,bool verbose){
  cube x = get_cube();
  if(verbose)
    std::cout << field << ": " << size(x) << std::endl;
  return x;
}

