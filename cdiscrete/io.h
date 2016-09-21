#ifndef __DATA_IO_INCLUDED__
#define __DATA_IO_INCLUDED__

#include <string>
#include <armadillo>

#include <archive.h>
#include <archive_entry.h>

using namespace std;
using namespace arma;


struct Archiver{
  bool add_mat(const string & field_name,
	   const mat & A);
  bool add_sp_mat(const string & field_name,
	   const sp_mat & A);
  bool add_vec(const string & field_name,
	   const vec & v);
  bool add_uvec(const string & field_name,
                const uvec & v);
  template <typename D>
  bool generic_add(const string & name,
		   const Col<D> & data);
  void write(const string & archive_name);
  vector<string> m_names;
  vector<string> m_data;
  
};

struct Unarchiver{
  Unarchiver(const string & archive_name);
  sp_mat load_sp_mat(const string & field_name);
  mat load_mat(const string & field_name);
  vec load_vec(const string & field_name);
  
  template <typename D>
  bool generic_load(const string & filename,
		    Col<D> & ret);
  archive* open_archive() const;
  string m_archive_name;
};
ostream& operator<< (ostream& os, const Unarchiver& unarch);


// Marshalling and unmarshalling matrices into vectors w/ headers
template<typename D> Col<D> pack_vec(const Col<D> &);
template<typename D> Col<D> pack_mat(const Mat<D> &);
template<typename D> Col<D> pack_sp_mat(const SpMat<D> &);

template<typename D> Col<D> unpack_vec(const Col<D> &);
template<typename D> Mat<D> unpack_mat(const Col<D> &);
template<typename D> SpMat<D> unpack_sp_mat(const Col<D> &);

#endif
