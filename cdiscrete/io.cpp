#include<assert.h>

#include "io.h"


void import_data(std::string dirname,
		 mat & q,
		 mat & flow,
		 RegGrid & grid){
  // Load the solver data
  // G x (2*A) matrix, where the A columns are the Q vector
  // and the rest are flow vectors
  string solver_data_filename = dirname + "data.h5";

  mat data;
  data.load(solver_data_filename,hdf5_binary);
  data = data.t();
  uint G = data.n_rows;
  uint A2 = data.n_cols;
  assert(0 == A2 % 2);
  uint A = A2 / 2;

  q = data.head_cols(A);
  flow = data.tail_cols(A);

  // Load the discretization data
  // <D,Lo[0],...,Lo[D-1],Hi[0],...,Hi[D-1],Num[0],...,Num[D-1]>
  string discrete_filename = dirname + "grid.h5";
  rowvec disc;
  disc.load(discrete_filename,hdf5_binary);
  std::cout<<size(disc)<<std::endl;
  uint D = disc(0);
  assert(disc.n_elem == 1 + 3*D);

  grid.low = disc( span(1,    1*D)).t();
  grid.high = disc(span(1+D,  2*D)).t();
  vec dnum = disc(span(1+2*D,3*D)).t();
  grid.num_cells = conv_to<uvec>::from(dnum);
  assert(G == prod(grid.num_cells + 1)+1);
}
