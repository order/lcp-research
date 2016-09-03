#include <iostream>
#include <string>
#include <sstream>
#include <assert.h>

#include <sys/types.h>

#include <sys/stat.h>

#include <archive.h>
#include <archive_entry.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <armadillo>
#include "io.h"

using namespace std;
using namespace arma;

void write_archive(const string & archive_name,
		     const vector<string> & field_names,
		     const vector<string> & buffers) {
  assert(field_names.size() == buffers.size());

  struct archive *a;
  struct archive_entry *entry;
  struct stat st;
  char buff[8192];
  int len;
  int fd;

  a = archive_write_new();
  archive_write_add_filter_gzip(a);
  archive_write_set_format_pax_restricted(a); // Note 1
  archive_write_open_filename(a, archive_name.c_str());

  typedef vector<string>::const_iterator iter;
  iter name_it = field_names.begin();
  iter buff_it = buffers.begin();
  uint sz;
  for(;name_it != field_names.end();){
    sz = buff_it->size();
    entry = archive_entry_new(); // Note 2
    archive_entry_set_pathname(entry, name_it->c_str());
    archive_entry_set_size(entry, sz); // Note 3
    archive_entry_set_filetype(entry, AE_IFREG);
    archive_entry_set_perm(entry, 0644);
    archive_write_header(a, entry);
    archive_write_data(a,buff_it->c_str(),sz);
    archive_entry_free(entry);
    ++name_it;
    ++buff_it;
  }
  archive_write_close(a); // Note 4
  archive_write_free(a); // Note 5
}


int main(int argc, char** argv)
{
  
  sp_mat A = sp_mat(randu<mat>(5,5));
  cout << "Packing...";
  vec packed_A = pack_sp_mat<double>(A);
  cout << "." << endl;
  cout << "Unpacking...";
  sp_mat B = unpack_sp_mat<double>(packed_A);
  cout << "." << endl;

  assert(approx_equal(A,B,"absdiff",1e-15));

  cout << "Sending to stream..." << endl;
  stringstream ss;
  packed_A.save(ss,raw_binary);
  cout << "Buffer: " << ss.str() << endl;
  vec read_in_A;
  read_in_A.load(ss,raw_binary);

  sp_mat C = unpack_sp_mat<double>(read_in_A);
  cout << "Read in:" << endl << C;
  assert(approx_equal(A,C,"absdiff",1e-15));

  vector<string> field_names,buffers;
  field_names.push_back("A.mat");
  buffers.push_back(ss.str());
  
  write_archive("test.tar.gz",field_names,buffers);
}
