#include<assert.h>
#include<armadillo>

#include "io.h"

#include <boost/filesystem.hpp>
#include <boost/filesystem/convenience.hpp>

/*
bool mkdir(const string & dir_path) {
  boost::filesystem::path dir(dir_path);
  return boost::filesystem::create_directories(dir);
}

int rmdir(const string &dir_path){
  boost::filesystem::path dir(dir_path);
  return boost::filesystem::remove_all(dir);
}

string strip_ext(const string & filname){
  return boost::filesystem::change_extension(filename, "").string();
}


Archiver::Archiver(const string & archive_name) : m_archive_name(archive_name){
  m_tmp_dir = "tmp_" + strip_ext(archive_name) + "_dir";
  rmdir(m_tmp_dir);
  mkdir(m_tmp_dir);
}

Archiver::~Archiver(){
}
*/

// Packing and unpacking to vectors

template<typename D> Col<D> pack_vec(const Col<D> & A){
  Col<D> ret = Col<D>(1 + A.n_elem);
  ret(0) = A.n_elem;
  ret.tail(A.n_elem) = vectorise(A);
  return ret;
}
template uvec pack_vec(const uvec &);
template vec pack_vec(const vec &);

template<typename D> Col<D> unpack_vec(const Col<D> & A){
  assert(A.n_elem >= 1);
  uint N = A(0);
  assert(1 + N == A.n_elem);
  return reshape(A.tail(N),size(N,1));
}
template uvec unpack_vec(const uvec &);
template vec unpack_vec(const vec &);

template<typename D> Col<D> pack_mat(const Mat<D> & A){
  Col<D> ret = Col<D>(2 + A.n_elem);
  ret(0) = A.n_rows;
  ret(1) = A.n_cols;
  ret.tail(A.n_elem) = vectorise(A);
  return ret;
}
template uvec pack_mat(const umat &);
template vec pack_mat(const mat &);

template<typename D> Mat<D> unpack_mat(const Col<D> & A){
  assert(A.n_elem >= 2);
  uint R = A(0);
  uint C = A(1);
  assert(2 + R*C == A.n_elem);
  return reshape(A.tail(R*C),size(R,C));
}
template umat unpack_mat(const uvec &);
template mat unpack_mat(const vec &);

template<typename D> Col<D> pack_sp_mat(const SpMat<D> & A){

  uint R = A.n_rows;
  uint C = A.n_cols;
  uint nnz = A.n_nonzero;

  uint header = 3;
  Col<D> data = Col<D>(header + 3*nnz);
  data(0) = R;
  data(1) = C;
  data(2) = nnz;

  typedef sp_mat::const_iterator SpIter;
  uint idx = 3;
  for(SpIter it = A.begin(); it != A.end(); ++it){
    data[idx++] = it.row();
    data[idx++] = it.col();
    data[idx++] = (*it);
  }
  assert(idx == data.n_elem);
  return data;
}
template vec pack_sp_mat(const sp_mat &);


template<typename D> SpMat<D> unpack_sp_mat(const Col<D> & A){
  assert(3 <= A.n_elem);
  uint R = A(0);
  uint C = A(1);
  uint nnz = A(2);
  assert((3 + 3*nnz) == A.n_elem);

  Mat<D> triples = reshape(A.tail(3*nnz),size(3,nnz));
  umat loc = conv_to<umat>::from(triples.head_rows(2));
  vec data = triples.row(2).t();
  return sp_mat(loc,data,R,C);
}
template sp_mat unpack_sp_mat(const vec &);
