#ifndef __DATA_IO_INCLUDED__
#define __DATA_IO_INCLUDED__

#include <string>
#include <armadillo>

#include "discrete.h"

using namespace arma;

/*
struct Archiver{
  Archiver(const string & archive_name);
  ~Archiver();
  
  template <typename D> bool add(const Mat<D> &);
  void write();

  string m_archive_name;
  string m_tmp_dir;
  vector<string> m_content_name;
  archive * m_archive_ptr;
};

bool mkdir(const string &);
string strip_ext(const string &);
*/




// Marshalling and unmarshalling matrices into vectors w/ headers
template<typename D> Col<D> pack_vec(const Col<D> &);
template<typename D> Col<D> pack_mat(const Mat<D> &);
template<typename D> Col<D> pack_sp_mat(const SpMat<D> &);

template<typename D> Col<D> unpack_vec(const Col<D> &);
template<typename D> Mat<D> unpack_mat(const Col<D> &);
template<typename D> SpMat<D> unpack_sp_mat(const Col<D> &);

#endif
