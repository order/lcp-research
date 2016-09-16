#include<assert.h>
#include<armadillo>

#include "io.h"

#include <boost/filesystem.hpp>
#include <boost/filesystem/convenience.hpp>

bool Archiver::add_mat(const string & field_name,
		       const mat & A){
  string name = field_name + ".mat";
  vec packed = pack_mat(A);
  generic_add(name,packed);
}

bool Archiver::add_sp_mat(const string & field_name,
		       const sp_mat & A){
  string name = field_name + ".sp_mat";
  vec packed = pack_sp_mat(A);
  generic_add(name,packed);
}

bool Archiver::add_vec(const string & field_name,
		       const vec & v){
  string name = field_name + ".vec";
  vec packed = pack_vec(v);
  generic_add(name,packed);
}

bool Archiver::add_uvec(const string & field_name,
		       const uvec & v){
  string name = field_name + ".uvec";
  uvec packed = pack_vec(v);
  generic_add(name,packed);
}

template <typename D>
bool Archiver::generic_add(const string & name,
			   const Col<D> & data){
    m_names.push_back(name);
    
    stringstream ss;
    data.save(ss,raw_binary);
    m_data.push_back(ss.str());
}

void Archiver::write(const string & archive_name) {
  assert(m_names.size() == m_data.size());

  struct archive *a;
  struct archive_entry *entry;

  a = archive_write_new();
  archive_write_add_filter_gzip(a);
  archive_write_set_format_pax_restricted(a); // Note 1
  archive_write_open_filename(a, archive_name.c_str());

  typedef vector<string>::const_iterator iter;
  iter name_it = m_names.begin();
  iter buff_it = m_data.begin();
  uint sz;
  for(;name_it != m_names.end();){
    sz = buff_it->size();
    entry = archive_entry_new();
    archive_entry_set_pathname(entry, name_it->c_str());
    archive_entry_set_size(entry, sz);
    archive_entry_set_filetype(entry, AE_IFREG);
    archive_entry_set_perm(entry, 0644);
    archive_write_header(a, entry);
    archive_write_data(a,buff_it->c_str(),sz);
    archive_entry_free(entry);
    ++name_it;
    ++buff_it;
  }
  archive_write_close(a);
  archive_write_free(a);
}

Unarchiver::Unarchiver(const string & archive_name){
  m_archive_ptr = archive_read_new();
  archive_read_support_format_all(m_archive_ptr);
  archive_read_support_filter_all(m_archive_ptr);

  int rc;
  if(rc = archive_read_open_filename(m_archive_ptr,
				     archive_name.c_str(),
				     16384)){
    cout << "Cannot open " << archive_name
	 << " (code " << rc << ")" << endl;
    exit(1);
  }  
}


mat Unarchiver::load_mat(const string & field_name){
  string filename = field_name + ".mat";
  vec raw_vec;
  bool rc = generic_load<double>(filename,raw_vec);
  assert(rc);
  return unpack_mat(raw_vec);
}
vec Unarchiver::load_vec(const string & field_name){
  string filename = field_name + ".vec";
  vec raw_vec;
  bool rc = generic_load<double>(filename,raw_vec);
  assert(rc);
  return unpack_vec(raw_vec);
}

// Split out mat specific and generic stuff.
template <typename D>
bool Unarchiver::generic_load(const string & filename,
				Col<D> & ret){
  archive_entry *entry;
  uint rc;
  char data[65536];
  size_t len;
  stringstream ss;
  
  while(true) {
    // Read entry
    rc = archive_read_next_header(m_archive_ptr, &entry);
    if(rc == ARCHIVE_EOF){
      return false;
    }
    if (rc != ARCHIVE_OK){
      cerr << "[ACHIVE ERROR]" << archive_error_string(m_archive_ptr) << endl;
      return false;
    }

    // Check header for a match
    if(strcmp(filename.c_str(),
	      archive_entry_pathname(entry))){
      continue; // No match, continue
    }

    // Buffered read to stringstream
    len = archive_read_data(m_archive_ptr,data,sizeof(data));
    while(len > 0){
      ss.write(data,len);
      len = archive_read_data(m_archive_ptr,data,sizeof(data));
    }

    // Success! Load the stringstream into the return vector
    ret.load(ss,raw_binary);
    return true;
  }
  // Should never hit here.
  assert(false);
}

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
