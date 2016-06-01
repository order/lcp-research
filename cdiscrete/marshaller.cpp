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

void Marshaller::save(const std::string & filename) const{
  vec output = vec(_header.n_elem + _data.n_elem);
  output.head(_header.n_elem) = conv_to<vec>::from(_header);
  output.tail(_data.n_elem) = _data;

  output.save(filename);
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
  std::cout << "Reading scalar..." << std::endl;
  assert(0 == _header[h_idx]);
  h_idx++;
  
  return _data(d_idx++);
}

vec Demarshaller::get_vec(){
  assert(1 == _header[h_idx]); // Should start on dim
  uint N = _header[++h_idx]; // Get len
  h_idx++; // Increment to next start

  std::cout << "Reading (" << N << ",) vector..." << std::endl;

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

  std::cout << "Reading (" << R << ","
	    << C << ") matrix..." << std::endl;

  vec v = _data(span(d_idx,d_idx+R*C-1));
  assert(R*C == v.n_elem);
  d_idx += R*C;
  
  return reshape(v,R,C);
}
