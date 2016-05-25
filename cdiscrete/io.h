#ifndef __DATA_IO_INCLUDED__
#define __DATA_IO_INCLUDED__

#include <armadillo>
using namespace arma;

#define IO_DIM_LOC 0;
#define IO_DIM_LOC 0;

void import_data(std::string dirname){
  // Load the solver data
  // G x (A+1) matrix, where the first column is the V vector
  // and the rest are flow vectors
  string solver_data_filename = dirname + ".solver_data.h5";
  mat solver_data;
  solver_data.load(solver_data_filename,hdf5_binary);
  uint G = solver_data.n_rows;
  uint A = solver_data.n_cols - 1;


  // Load the discretization data
  // <D,Lo[0],...,Lo[D-1],Hi[0],...,Hi[D-1],Num[0],...,Num[D-1]>
  string discrete_filename = dirname + ".discrete.h5";
  vec disc;
  disc.load(discrete_filename,hdf5_binary);
  uint D = disc(0);
  assert(disc.n_elem == 1 + 3*D);

  vec low = disc( span(1,    1*D));
  vec high = disc(span(1+D,  2*D));
  vec dnum = disc(span(1+2*D,3*D));
  uvec num_cells = conv_to<uvec>::from(dnum);

  assert(G == prod(num_cells + 1)+1);
}

#endif
